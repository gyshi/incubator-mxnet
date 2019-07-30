/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2019 by Contributors
 * \file broadcast_reduce-inl.cuh
 * \brief CUDA implementations for linalg norm
*/

#include "../tensor/broadcast_reduce-inl.cuh"

#ifndef MXNET_OPERATOR_NUMPY_NP_LA_OP_CUH_
#define MXNET_OPERATOR_NUMPY_NP_LA_OP_CUH_

using namespace mshadow::cuda;

template<typename Reducer, int ndim, typename DType, typename OP1, typename OP2>
__global__ void power_reduce_kernel_M1(const int N, const bool addto,
                                 const DType* __restrict big,
                                 const DType* __restrict lhs,
                                 const DType* __restrict rhs,
                                 DType *small,
                                 const Shape<ndim> big_shape,
                                 const Shape<ndim> lhs_shape,
                                 const Shape<ndim> rhs_shape,
                                 const Shape<ndim> small_shape,
                                 const float ord) {
  for (int idx = threadIdx.x + blockIdx.x*blockDim.x; idx < N; idx += blockDim.x*gridDim.x) {
    Shape<ndim> coord = unravel(idx, small_shape);
    int idx_big = ravel(coord, big_shape);
    int idx_lhs = ravel(coord, lhs_shape);
    int idx_rhs = ravel(coord, rhs_shape);
    DType val, residual;
    residual = DType(ord);
    Reducer::SetInitValue(val, residual);
    Reducer::Reduce(val, OP1::Map(big[idx_big], OP2::Map(lhs[idx_lhs], rhs[idx_rhs])), residual);
    Reducer::Finalize(val, residual);
    assign(&small[idx], addto, val);
  }
}


#define KERNEL_UNROLL_SWITCH(do_unroll, unrollAmount, unrollVar, ...) \
  if (do_unroll) {                                                    \
    const int unrollVar = unrollAmount;                               \
    {__VA_ARGS__}                                                     \
  } else {                                                            \
    const int unrollVar = 1;                                          \
    {__VA_ARGS__}                                                     \
  }

template<typename Reducer, int ndim, typename AType, typename DType, typename OType, typename OP>
void PowerReduceImpl(cudaStream_t stream, const TBlob& small, const OpReqType req,
                const TBlob& big, const Tensor<gpu, 1, char>& workspace,
                const ReduceImplConfig<ndim>& config, const float ord) {
  if (config.M == 1) {
    power_reduce_kernel_M1<Reducer, ndim, AType, DType, OType, OP>
        <<< config.kernel_1.gridDim, config.kernel_1.blockDim, 0, stream >>>(
        config.N, req == kAddTo, big.dptr<DType>(), small.dptr<OType>(), big.shape_.get<ndim>(),
            small.shape_.get<ndim>(), ord);
    MSHADOW_CUDA_POST_KERNEL_CHECK(power_reduce_kernel_M1);
  } else {
    OType* small_dptr = small.dptr<OType>();
    bool addto = (req == kAddTo);
    if (config.Mnext > 1) {
      // small_dptr[] is N*Mnext*sizeof(DType) bytes
      small_dptr = reinterpret_cast<OType*>(workspace.dptr_);
      addto = false;
      // Check that the workspace is contigiuous
      CHECK_EQ(workspace.CheckContiguous(), true);
      // Check that we have enough storage
      CHECK_GE(workspace.size(0), config.workspace_size);
    }

    const int by = (config.kernel_1.do_transpose) ?
                   config.kernel_1.blockDim.x : config.kernel_1.blockDim.y;
    const bool do_unroll = ( config.M / (by*config.Mnext) >= config.unroll_reduce );
    KERNEL_UNROLL_SWITCH(do_unroll, ReduceImplConfig<ndim>::unroll_reduce, UNROLL, {
      reduce_kernel<Reducer, ndim, AType, DType, OType, OP, UNROLL>
      <<< config.kernel_1.gridDim, config.kernel_1.blockDim, config.kernel_1.shMemSize, stream>>>(
          config.N, config.M, addto, big.dptr<DType>(), small_dptr, big.shape_.get<ndim>(),
              small.shape_.get<ndim>(), config.rshape, config.rstride, config.Mnext,
              config.kernel_1.do_transpose);
    });
    MSHADOW_CUDA_POST_KERNEL_CHECK(reduce_kernel);

    if (config.Mnext > 1) {
      reduce_lines_kernel<Reducer, OType>
      <<< config.kernel_2.gridSize, config.kernel_2.blockSize, 0, stream >>>
          (config.N, config.Mnext, req == kAddTo, config.N, small_dptr, small.dptr<OType>());
      MSHADOW_CUDA_POST_KERNEL_CHECK(reduce_lines_kernel);
    }
  }
}
#undef KERNEL_UNROLL_SWITCH

template<typename Reducer, int ndim, typename DType, typename OP, bool safe_acc = false>
void PowerReduce(Stream<gpu> *s, const TBlob& small, const OpReqType req,
            const Tensor<gpu, 1, char>& workspace, const TBlob& big, const float ord) {
  if (req == kNullOp) return;
  cudaStream_t stream = Stream<gpu>::GetStream(s);
  ReduceImplConfig<ndim> config =
      ConfigureReduceImpl<ndim, DType>(small.shape_, big.shape_, NULL, NULL);
  if (safe_acc) {
    // TODO(haojin2): Use real-only type swtich for windows temporarily due to CI issues.
#ifndef _WIN32
    MXNET_ACC_TYPE_SWITCH(mshadow::DataType<DType>::kFlag, DataType, AType, {
        typedef typename std::conditional<safe_acc, AType, DataType>::type AccType;
        MSHADOW_TYPE_SWITCH(small.type_flag_, OType, {
          typedef typename std::conditional<safe_acc, OType, DataType>::type OutType;
          config = ConfigureReduceImpl<ndim, AccType>(small.shape_, big.shape_, NULL, NULL);
          PowerReduceImpl<Reducer, ndim, AccType, DataType, OutType, OP>(
              stream, small, req, big, workspace, config, ord);
        });
    });
#else
    MXNET_REAL_ACC_TYPE_SWITCH(mshadow::DataType<DType>::kFlag, DataType, AType, {
      typedef typename std::conditional<safe_acc, AType, DataType>::type AccType;
      MSHADOW_TYPE_SWITCH(small.type_flag_, OType, {
        typedef typename std::conditional<safe_acc, OType, DataType>::type OutType;
        config = ConfigureReduceImpl<ndim, AccType>(small.shape_, big.shape_, NULL, NULL);
        PowerReduceImpl<Reducer, ndim, AccType, DataType, OutType, OP>(
          stream, small, req, big, workspace, config, ord);
      });
    });
#endif
  } else {
    PowerReduceImpl<Reducer, ndim, DType, DType, DType, OP>(stream, small, req, big, workspace, config, ord);
  }
}

#endif  //MXNET_OPERATOR_NUMPY_NP_LA_OP_CUH_
