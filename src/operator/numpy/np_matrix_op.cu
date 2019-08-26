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
 *  Copyright (c) 2019 by Contributors
 * \file np_matrix_op.cu
 * \brief GPU Implementation of numpy matrix operations
 */

#include "./np_matrix_op-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_np_transpose)
.set_attr<FCompute>("FCompute<gpu>", NumpyTranspose<gpu>);

NNVM_REGISTER_OP(_np_reshape)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(_np_squeeze)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(_npi_concatenate)
.set_attr<FCompute>("FCompute<gpu>", ConcatCompute<gpu>);

NNVM_REGISTER_OP(_backward_np_concat)
.set_attr<FCompute>("FCompute<gpu>", ConcatGradCompute<gpu>);

NNVM_REGISTER_OP(_npi_stack)
.set_attr<FCompute>("FCompute<gpu>", StackOpForward<gpu>);

NNVM_REGISTER_OP(_npi_hsplit)
.set_attr<FCompute>("FCompute<gpu>", HSplitOpForward<gpu>);

NNVM_REGISTER_OP(_npi_hsplit_backward)
.set_attr<FCompute>("FCompute<gpu>", HSplitOpBackward<gpu>);

}  // namespace op
}  // namespace mxnet
