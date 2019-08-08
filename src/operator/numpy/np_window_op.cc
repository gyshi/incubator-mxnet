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
 * \file np_window_op.cc
 * \brief CPU Implementation of unary op hanning, hamming, blackman window.
 */

#include "np_window_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(NumpyWindowsParam);

NNVM_REGISTER_OP(_npi_blackman)
.describe("Return the Blackman window."
"The Blackman window is a taper formed by using a weighted cosine.")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NumpyWindowsParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyWindowsShape)
.set_attr<nnvm::FInferType>("FInferType", InitType<NumpyWindowsParam>)
.set_attr<FCompute>("FCompute<cpu>", NumpyWindowCompute<cpu, 2>)
.add_arguments(NumpyWindowsParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet