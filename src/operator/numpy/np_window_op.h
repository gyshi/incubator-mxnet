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
 * \file np_window_op.h
 * \brief CPU Implementation of unary op hanning, hamming, blackman window.
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_WINDOW_OP_H_
#define MXNET_OPERATOR_NUMPY_NP_WINDOW_OP_H_

#include <string>
#include <vector>
#include "./np_init_op.h"

namespace mxnet {
namespace op {

#ifdef __CUDA_ARCH__
__constant__ const float PI = 3.14159265358979323846;
#else
const float PI = 3.14159265358979323846;
using std::isnan;
#endif

struct NumpyWindowsParam : public dmlc::Parameter<NumpyWindowsParam> {
  dmlc::optional<nnvm::dim_t> M;
  std::string ctx;
  int dtype;
  DMLC_DECLARE_PARAMETER(NumpyWindowsParam) {
      DMLC_DECLARE_FIELD(M)
      .set_default(dmlc::optional<nnvm::dim_t>())
      .describe("Number of points in the output window. "
                "If zero or less, an empty array is returned.");
      DMLC_DECLARE_FIELD(ctx)
      .set_default("")
      .describe("Context of output, in format [cpu|gpu|cpu_pinned](n)."
      "Only used for imperative calls.");
      DMLC_DECLARE_FIELD(dtype)
      .set_default(mshadow::kFloat64)
      MXNET_ADD_ALL_TYPES
      .describe("Data-type of the returned array.");
  }
};

inline bool NumpyWindowsShape(const nnvm::NodeAttrs& attrs,
                            mxnet::ShapeVector* in_shapes,
                            mxnet::ShapeVector* out_shapes) {
  const NumpyWindowsParam& param = nnvm::get<NumpyWindowsParam>(attrs.parsed);
  CHECK_EQ(in_shapes->size(), 0U);
  CHECK_EQ(out_shapes->size(), 1U);
  CHECK(param.M.has_value()) << "missing 1 required positional argument: 'M'";
  int64_t out_size = param.M.value() <= 0 ? 0 : param.M.value();
  SHAPE_ASSIGN_CHECK(*out_shapes, 0, mxnet::TShape({static_cast<nnvm::dim_t>(out_size)}));
  return true;
}

struct windows_fwd {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, index_t M, index_t window_select,
                                  int req, DType* out) {
    // hanning window
    if (window_select == 0) {
      if (M == 1) {
        KERNEL_ASSIGN(out[i], req, static_cast<int64_t>(1));
      } else {
        KERNEL_ASSIGN(out[i], req,  DType(0.5)-DType(0.5)*math::cos(DType(2*PI*i/(M-1))));
      }
    } else if (window_select == 1) {  // hamming window
      if (M == 1) {
        KERNEL_ASSIGN(out[i], req, static_cast<int64_t>(1));
      } else {
        KERNEL_ASSIGN(out[i], req,  DType(0.54)-DType(0.46)*math::cos(DType(2*PI*i/(M-1))));
      }
    } else if (window_select == 2) {  // blackman window
      if (M == 1) {
        KERNEL_ASSIGN(out[i], req, static_cast<int64_t>(1));
      } else {
        KERNEL_ASSIGN(out[i], req,  DType(0.42) -
                      DType(0.5) * math::cos(DType(2 * PI * i / M)) +
                      DType(0.08) * math::cos(DType(4 * PI * i / M)));
      }
    } else {
      LOG(FATAL) << "window_select must be (0, 1, 2)";
    }
  }
};

template<typename xpu, int window_select>
void NumpyWindowCompute(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<mxnet::TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const NumpyWindowsParam& param = nnvm::get<NumpyWindowsParam>(attrs.parsed);

  if (param.M.has_value() && param.M.value() <= 0) return;
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        Kernel<windows_fwd, xpu>::Launch(s,
                                       outputs[0].Size(),
                                       static_cast<int>(param.M.value()),
                                       static_cast<int>(window_select),
                                       req[0],
                                       outputs[0].dptr<DType>());
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_WINDOW_OP_H_
