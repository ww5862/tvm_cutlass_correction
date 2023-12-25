# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name
"""GEMM kernel generator and profiler for CUTLASS."""
from .profile_cutlass import ProfileGemm
from .gemm_operation import EmitGemmInstance, GemmOperation
from .gemm_profiler import GemmProfilerEmitter
from .gen_tensor_op import EPILOGUE_MAP, GENERATOR_FUNC_TABLE, ProfilerEngine
from .library import (
    DataType,
    DataTypeTag,
    EpilogueFunctor,
    LayoutType,
    SwizzlingFunctor,
    TensorDescription,
    TileDescription,
    MathInstruction,
    DataType,
    OpcodeClass,
    MathOperation,
)

import json
import os


def create_gemm_operator_with_epilogue(
    op_type,
    tile_description,
    data_type,
    alignment,
    swizzling_functor,
    split_k=1,
    batched=False,
    transpose_a=False,
    transpose_b=True,
):
    """
    Instantiate a cutlass kernel from the given configuration,
    along with the epilouge functor
    """
    element_a, element_b, element_c, element_epilogue = data_type
    
    if transpose_a == True:
        A = TensorDescription(element_a, LayoutType.ColumnMajor, alignment)
    else:
        A = TensorDescription(element_a, LayoutType.RowMajor, alignment)
    
    if transpose_b == True:
        B = TensorDescription(element_b, LayoutType.ColumnMajor, alignment)
    else:
        B = TensorDescription(element_b, LayoutType.RowMajor, alignment)
    
    C = TensorDescription(element_c, LayoutType.RowMajor, alignment)
    
    if batched:
        swizzling_functor = SwizzlingFunctor.Batched

    epilogue, no_beta_scaling = EPILOGUE_MAP[op_type]

    op = GemmOperation(
        tile_description.minimum_compute_capability,
        tile_description,
        A,
        B,
        C,
        element_epilogue,
        epilogue,
        swizzling_functor,
    )
    
    return (
        op.procedural_name(),
        EmitGemmInstance().emit(op, no_beta_scaling=no_beta_scaling, split_k=split_k, batched=batched),
    )


def enumerate_gemm_operators(
    tile_descriptions,
    data_type,
    alignment_constraints,
    swizzling_functor=SwizzlingFunctor.Identity8,
    transpose_a=False,
    transpose_b=True,
):
    """Exhaustively instantiate all kernels from a given configuration."""
    ret = []
    kernel_emitter = EmitGemmInstance()
    profiler_emitter = GemmProfilerEmitter()

    element_a, element_b, element_c, element_epilogue = data_type

    for tile_description in tile_descriptions:
        for alignment in alignment_constraints:
            if transpose_a == True:
                A = TensorDescription(element_a, LayoutType.ColumnMajor, alignment)
            else:
                A = TensorDescription(element_a, LayoutType.RowMajor, alignment)
            
            if transpose_b == True:
                B = TensorDescription(element_b, LayoutType.ColumnMajor, alignment)
            else:
                B = TensorDescription(element_b, LayoutType.RowMajor, alignment)
            
            C = TensorDescription(element_c, LayoutType.RowMajor, alignment)
            
            if element_c == DataType.s32 and A.alignment == 1:
                tile_description.threadblock_shape[0] = min(
                    tile_description.threadblock_shape[0], 1024
                )
                tile_description.threadblock_shape[1] = min(
                    tile_description.threadblock_shape[1], 1024
                )

            op = GemmOperation(
                tile_description.minimum_compute_capability,
                tile_description,
                A,
                B,
                C,
                element_epilogue,
                EpilogueFunctor.LinearCombination,
                swizzling_functor,
            )

            src = profiler_emitter.emit(
                op.procedural_name(),
                kernel_emitter.emit(op, batched=False),
                DataTypeTag[element_a],
                DataTypeTag[element_b],
                DataTypeTag[element_c],
                op.leading_dim(),
            )
            
            ret.append(
                {
                    "src": src,
                    "op": op,
                    "name": op.procedural_name(),
                    "tile_description": tile_description,
                    "alignment": alignment,
                    "data_type": data_type,
                    "swizzle_functor": swizzling_functor,
                    "split_k": tile_description.split_k
                }
            )

    return ret


# TODO(masahi): A sensible way to pick reasonable default kernels
DEFAULT_KERNELS = {
    75: {
        ("float16", "float16"): "cutlass_tensorop_h1688gemm_128x64_32x2_tn_align1",
        ("float16", "float32"): "cutlass_tensorop_s1688gemm_f16_64x64_32x2_tn_align1",
    },
    # align1 variants do not seem to be available for sm80
    80: {
        ("float16", "float16"): "cutlass_tensorop_h1688gemm_128x64_32x2_tn_align1",
        ("float16", "float32"): "cutlass_tensorop_s1688gemm_f16_64x64_32x2_tn_align1",
        # two kernels for tf32 and 3xtf32
        ("float32", "float32"): (
            "cutlass_tensorop_s1688gemm_128x64_32x3_tn_align1",
            "cutlass_tensorop_s1688gemm_64x64_16x3_tn_align1",
        ),
    },
}


class CutlassGemmProfiler:
    """Profile all candidate kernels and select the best one."""

    def __init__(self, sm, cutlass_path, binary_path):
        assert sm in GENERATOR_FUNC_TABLE and sm in DEFAULT_KERNELS, "sm%d not supported yet." % sm
        self.engine = ProfilerEngine(sm, cutlass_path, binary_path)
        self.sm = sm
        self.cache = {}

    def get_default(
        self, op_type, out_dtype, arg0_dtype, arg1_dtype, use_3xtf32=True, batched=False
    ):
        """Return the default kernel for the requested architecture.
        For now, the default kernel was picked arbitrary.
        """
        ops = GENERATOR_FUNC_TABLE[self.sm](
            out_dtype,
            arg0_dtype,
            arg1_dtype,
            enumerate_gemm_operators,
            lambda align: align == 1,  # Only request align1 kernels
            use_3xtf32,
            profile_all_alignments=True,  # To include all align1 kernels
            # TODO(masahi): Invesitigate when fp32 accumulation is needed for gemm
            accumlator_dtype=out_dtype,
        )

        default_kernel_name = DEFAULT_KERNELS[self.sm][(arg0_dtype, out_dtype)]

        if arg0_dtype == "float32":
            default_kernel_name = (
                default_kernel_name[0] if not use_3xtf32 else default_kernel_name[1]
            )

        filtered = list(filter(lambda op: op["name"] == default_kernel_name, ops))
        assert len(filtered) == 1
        op = filtered[0]
        name, opdef = create_gemm_operator_with_epilogue(
            op_type,
            op["tile_description"],
            op["data_type"],
            op["alignment"],
            op["swizzle_functor"],
            batched=batched,
        )
        op.update({"name": name, "opdef": opdef})
        return op

    def select_op(
        self,
        M,
        N,
        K,
        out_dtype,
        arg0_dtype,
        arg1_dtype,
        use_3xtf32,
        profile_all_alignments=False,
        find_first_valid=False,
        use_multiprocessing=False,
        batch=False,
        batch_count=1,
        transpose_a=False,
        transpose_b=True,
        op_type="",
    ):
        """
        Profile and select the best kernel from candidate kernels.
        See the documentation for the profile method below.
        """
        if (M, N, K) in self.cache:
            op = self.cache[(M, N, K)]
            return op

        # TODO(masahi): CUTLASS alignment check on gemm kernels is too restrictive.
        # See https://github.com/NVIDIA/cutlass/issues/362.
        # When the above issue is resolved, we can remove the alignment check on M below.
        
        real_path = os.getcwd()
        
        # print(f"{op_type}: {M}, {N}, {K}")
        # dir_path = f"{real_path}/bert_dim.json"
        # if "batch" in op_type:
        #     json_data = {"model": "bert-large-uncased", "op_type": str(op_type), "dim": [int(batch_count), int(M), int(N), int(K)]}
        # else:
        #     json_data = {"model": "bert-large-uncased", "op_type": str(op_type), "dim": [int(M), int(N), int(K)]}
            
        # with open(dir_path, 'a') as f:
        #     json.dump(json_data, f)
        #     f.write("\n")
        
        #insert code for oracle
        if arg0_dtype == "float32" and arg1_dtype == "float32":
            if out_dtype == "float32":
                math_instruction = [
                    MathInstruction(
                        [1, 1, 1],
                        DataType.f32,
                        DataType.f32,
                        DataType.f32,
                        DataType.f32,
                        OpcodeClass.Simt,
                        MathOperation.multiply_add,
                    ),
                ]
                
                alignment_constraints = [1,]
                
                gemm_profile = ProfileGemm(batch=int(batch_count), M=int(M), N=int(N), K=int(K), split_k=16)
                tile, split = gemm_profile.eval_cutlassOracle(transpose_a=transpose_a, transpose_b=transpose_b)
                
                #set tile description here!
                print(f"{tile}, {split}")
                
                block_tile = tile[0]
                buffer_stage = tile[2][0]
                warp_tile = [int(tile[0][0] / tile[1][0]), int(tile[0][1] / tile[1][1]), int(tile[0][2] / tile[1][2])]
                
                
                tile_descriptions  = [(block_tile, buffer_stage, warp_tile, split, 80, 1024)]
                print(tile_descriptions)
                
                description_all = [
                    TileDescription(threadblock_shape, stages, warp_count, math_instruction[0], min_cc, max_cc, split_k=split_k)
                    for threadblock_shape, stages, warp_count, split_k, min_cc, max_cc in tile_descriptions
                ]
                
                data_dtype = [
                    math_instruction[0].element_a,
                    math_instruction[0].element_b,
                    math_instruction[0].element_c,
                    math_instruction[0].element_accumulator,
                ]
                
                out_ops = enumerate_gemm_operators(description_all, data_dtype, alignment_constraints, transpose_a=transpose_a, transpose_b=transpose_b)
                out_ops[0]["runtime"] = 0.0
                self.cache[(M, N, K)] = out_ops[0]
                
                return out_ops[0]
            #end of insert
        
        ops = GENERATOR_FUNC_TABLE[self.sm](
            out_dtype,
            arg0_dtype,
            arg1_dtype,
            enumerate_gemm_operators,
            lambda align: all([dim % align == 0 for dim in [M, N, K]]),
            use_3xtf32,
            profile_all_alignments=profile_all_alignments,
            # TODO(masahi): Invesitigate when fp32 accumulation is needed for gemm
            accumlator_dtype=out_dtype,
        )

        if not find_first_valid:
            self.engine.compile_all(ops, use_multiprocessing)

        for op in ops:
            out = self.engine.evaluate(op, [M, N, K])
            op["runtime"] = out
            if out < float("inf") and find_first_valid:
                self.cache[(M, N, K)] = op
                return op

        op = min(ops, key=lambda i: i["runtime"])
        self.cache[(M, N, K)] = op
        return op

    def profile(
        self,
        op_type,
        M,
        N,
        K,
        out_dtype,
        arg0_dtype,
        arg1_dtype,
        use_3xtf32=True,
        profile_all_alignments=False,
        find_first_valid=False,
        use_multiprocessing=False,
        batched=False,
        batch_count = 1,
        transpose_a = False,
        transpose_b = True,
    ):
        """Profile and select the best kernel from candidate kernels.
        If find_first_valid is True, return immediately after the first applicable kernel is found.
        If use_multiprocessing is True, compile all profiler executables in parallel.
        """
        
        op = self.select_op(
            M,
            N,
            K,
            out_dtype,
            arg0_dtype,
            arg1_dtype,
            use_3xtf32,
            profile_all_alignments=profile_all_alignments,
            find_first_valid=find_first_valid,
            use_multiprocessing=use_multiprocessing,
            batch=batched,
            batch_count=batch_count,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            op_type=op_type,
        )
        
        name, opdef = create_gemm_operator_with_epilogue(
            op_type,
            op["tile_description"],
            op["data_type"],
            op["alignment"],
            op["swizzle_functor"],
            split_k=op["split_k"],
            batched=batched,
            transpose_a=transpose_a,
            transpose_b=transpose_b
        )

        return name, opdef, op["runtime"], op["split_k"]
