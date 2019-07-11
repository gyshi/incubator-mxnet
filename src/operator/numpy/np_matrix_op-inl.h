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
 * \file np_matrix_op-inl.h
 * \brief Function definition of matrix related operators
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_MATRIX_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_MATRIX_OP_INL_H_

#include <vector>
#include "../tensor/matrix_op-inl.h"

namespace mxnet {
namespace op {

struct NumpyTransposeParam : public dmlc::Parameter<NumpyTransposeParam> {
  mxnet::TShape axes;
  DMLC_DECLARE_PARAMETER(NumpyTransposeParam) {
    DMLC_DECLARE_FIELD(axes).set_default(mxnet::TShape(-1, 0))
    .describe("By default, reverse the dimensions, otherwise permute "
              "the axes according to the values given.");
  }
};

template<typename xpu>
void NumpyTranspose(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  const NumpyTransposeParam& param = nnvm::get<NumpyTransposeParam>(attrs.parsed);
  CHECK_EQ(req[0], kWriteTo) << "Transpose does not support inplace";
  if (ndim_is_known(param.axes)) {
    TransposeImpl<xpu>(ctx.run_ctx, inputs[0], outputs[0], param.axes);
  } else {
    mxnet::TShape axes(inputs[0].ndim(), -1);
    for (int i = 0; i < axes.ndim(); ++i) {
      axes[i] = axes.ndim() - 1 - i;
    }
    TransposeImpl<xpu>(ctx.run_ctx, inputs[0], outputs[0], axes);
  }
}

struct NumpyMoveaxisParam : public dmlc::Parameter<NumpyMoveaxisParam> {
  mxnet::TShape sources;
  mxnet::TShape destination;
  DMLC_DECLARE_PARAMETER(NumpyMoveaxisParam) {
    DMLC_DECLARE_FIELD(sources).set_default(mxnet::TShape(-1, 0))
        .describe("By default, reverse the dimensions, otherwise permute "
                  "the axes according to the values given.");
    DMLC_DECLARE_FIELD(destination).set_default(mxnet::TShape(-1, 0))
        .describe("By default, reverse the dimensions, otherwise permute "
  "the axes according to the values given.");
}
};

template<typename xpu>
void NumpyMoveaxisOpForward(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const NumpyMoveaxisParam& param = nnvm::get<NumpyMoveaxisParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req[0], kWriteTo) << "Moveaxis does not support inplace";
  mxnet::TShape axes(inputs[0].ndim(), -1);
  std::vector<bool> state_axes(inputs[0].ndim(), false);
  for (int i = 0; i < param.sources.ndim(); ++i) {
    axes[param.destination[i]] = param.sources[i];
    state_axes[param.sources[i]] = true;
  }
  for (int i = 0; i < axes.ndim(); ++i) {
    if (axes[i] < 0) {
      for (int j = 0; j < axes.ndim(); ++j) {
        if (state_axes[j] == false) {
          axes[i] = j;
          state_axes[j] = true;
          break;
        }
      }
    }
  }
  TransposeImpl<xpu>(ctx.run_ctx, inputs[0], outputs[0], axes);
}

template<typename xpu>
void NumpyMoveaxisOpBackward(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const NumpyMoveaxisParam& param = nnvm::get<NumpyMoveaxisParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req[0], kWriteTo) << "Moveaxis does not support inplace";
  mxnet::TShape axes(inputs[0].ndim(), -1);
  std::vector<bool> state_axes(inputs[0].ndim(), false);
  for (int i = 0; i < param.sources.ndim(); ++i) {
    axes[param.sources[i]] = param.destination[i];
    state_axes[param.destination[i]] = true;
  }
  for (int i = 0; i < axes.ndim(); ++i) {
    if (axes[i] < 0) {
      for (int j = 0; j < axes.ndim(); ++j) {
        if (state_axes[j] == false) {
          axes[i] = j;
          state_axes[j] = true;
          break;
        }
      }
    }
  }
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, Dtype, {
    TransposeImpl<xpu>(ctx.run_ctx, inputs[0], outputs[0], axes);
  })

}

bool NumpyMoveaxisShape(const nnvm::NodeAttrs& attrs,
                         mxnet::ShapeVector *in_attrs,
                         mxnet::ShapeVector *out_attrs) {
  const NumpyMoveaxisParam& param = nnvm::get<NumpyMoveaxisParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  mxnet::TShape& shp = (*in_attrs)[0];
  CHECK_LE(shp.ndim(), 6) << "Transpose support at most 6 dimensions";
  CHECK_EQ(param.sources.ndim(), param.destination.ndim()) \
        << "sources and destination not equal.";
  mxnet::TShape ret(shp.ndim(), -1);
  mxnet::TShape axes(shp.ndim(), -1);
  std::vector<bool> state_axes(shp.ndim(), false);
  for (int i = 0; i < param.sources.ndim(); ++i) {
    axes[param.destination[i]] = param.sources[i];
    state_axes[param.sources[i]] = true;
  }
  for (int i = 0; i < axes.ndim(); ++i) {
    if (axes[i] < 0) {
      for (int j = 0; j < axes.ndim(); ++j) {
        if (state_axes[j] == false) {
          axes[i] = j;
          state_axes[j] = true;
          break;
        }
      }
    }
  }
  for (int i = 0; i < shp.ndim(); ++i) {
      CHECK(axes[i] < static_cast<int64_t>(shp.ndim()));
      ret[i] = shp[axes[i]];
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, ret);
  return shape_is_known(ret);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_MATRIX_OP_INL_H_
