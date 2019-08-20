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
 * \file ufunc.cc
 * \brief
 * \author Yizhi Liu
 */
#ifdef MXNET_USE_TVM_OP
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>
#include <mxnet/base.h>
#include "../../tensor/elemwise_binary_broadcast_op.h"
#include "../../tvmop/op_module.h"
#include "../../tensor/elemwise_binary_op.h"

namespace mxnet {
namespace op {

static constexpr char func_vadd_cpu[] = "vadd";
static constexpr char func_vadd_gpu[] = "cuda_vadd";

template<const char* func>
void TVMBroadcastCompute(const nnvm::NodeAttrs& attrs,
                         const mxnet::OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  tvm::runtime::TVMOpModule::Get()->Call(func, ctx, {inputs[0], inputs[1], outputs[0]});
}

NNVM_REGISTER_OP(_contrib_tvm_vadd)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .add_argument("a", "NDArray-or-Symbol", "first input")
    .add_argument("b", "NDArray-or-Symbol", "second input")
    .set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)
    .set_attr<nnvm::FInferType>("FInferType", mxnet::op::ElemwiseType<2, 1>)
    .set_attr<mxnet::FCompute>("FCompute<cpu>", mxnet::op::TVMBroadcastCompute<func_vadd_cpu>)
#if MXNET_USE_CUDA
    .set_attr<mxnet::FCompute>("FCompute<gpu>", mxnet::op::TVMBroadcastCompute<func_vadd_gpu>);
#endif  // MXNET_USE_CUDA

static constexpr char func_exp2_cpu[] = "exp2";
static constexpr char func_epx2_gpu[] = "cuda_exp2";
static constexpr char func_backward_epx2_cpu[] = "backward_exp2";
static constexpr char func_backward_epx2_gpu[] = "cuda_backward_exp2";

template<const char* func>
void TVMOpExp2Compute(const nnvm::NodeAttrs& attrs,
                      const mxnet::OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  tvm::runtime::TVMOpModule::Get()->Call(func, ctx, {inputs[0], outputs[0]});
}

//template<int req>
//struct exp2_backward {
//  template<typename DType>
//  MSHADOW_XINLINE static void Map(int i, DType* in_grad, const DType* in_data1,
//                                  const DType* in_data2, const DType* out_grad) {
//    DType grad = in_data2[i] * in_data1[i] / DType(2);
//    KERNEL_ASSIGN(in_grad[i], req, grad);
//  }
//};

template<const char* func>
void TVMExp2Backward(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  using namespace mshadow;
  const TBlob& out_grad = inputs[0];
  const TBlob& in_data1 = inputs[1];
  const TBlob& in_data2 = inputs[2];
  const TBlob& in_grad = outputs[0];
  tvm::runtime::TVMOpModule::Get()->Call(func, ctx, {out_grad, in_data1, in_data2, in_grad});
}

NNVM_REGISTER_OP(_contrib_tvm_exp2)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .add_argument("data", "NDArray-or-Symbol", "resources")
    .set_attr<mxnet::FInferShape>("FInferShape", mxnet::op::ElemwiseShape<1, 1>)
    .set_attr<nnvm::FInferType>("FInferType", mxnet::op::ElemwiseType<1, 1>)
#if MXNET_USE_CUDA
    .set_attr<mxnet::FCompute>("FCompute<gpu>", mxnet::op::TVMOpExp2Compute<func_epx2_gpu>)
#endif  // MXNET_USE_CUDA
    .set_attr<mxnet::FCompute>("FCompute<cpu>", mxnet::op::TVMOpExp2Compute<func_exp2_cpu>)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseInOut{ "_tvm_backward_exp2" });



NNVM_REGISTER_OP(_tvm_backward_exp2)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
#if MXNET_USE_CUDA
.set_attr<FCompute>("FCompute<gpu>", mxnet::op::TVMExp2Backward<func_backward_epx2_gpu>)
#endif  // MXNET_USE_CUDA
.set_attr<FCompute>("FCompute<cpu>", mxnet::op::TVMExp2Backward<func_backward_epx2_cpu>);

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_TVM_OP
