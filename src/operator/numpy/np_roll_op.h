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
 * \file np_roll_op.h
 * \brief CPU Implementation of numpy rearranging elements operator
 * \author Guangyong Shi
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_ROLL_OP_CC_H_
#define MXNET_OPERATOR_NUMPY_NP_ROLL_OP_CC_H_

#include <vector>
#include <mshadow/tensor.h>
#include "../tensor/init_op.h"

namespace mxnet {
namespace op {

using namespace mshadow;

struct NumpyRollParam : public dmlc::Parameter<NumpyRollParam> {
  dmlc::optional<mxnet::TShape> shift;
  dmlc::optional<mxnet::TShape> axis;
  DMLC_DECLARE_PARAMETER(NumpyRollParam) {
    DMLC_DECLARE_FIELD(shift)
    .set_default(dmlc::optional<mxnet::TShape>())
    .describe("The number of places by which elements are shifted. If a tuple,"
              "then axis must be a tuple of the same size, and each of the given axes is shifted"
              "by the corresponding number. If an int while axis is a tuple of ints, "
              "then the same value is used for all given axes.");
    DMLC_DECLARE_FIELD(axis)
    .set_default(dmlc::optional<mxnet::TShape>())
    .describe("Axis or axes along which elements are shifted. By default, "
    "the array is flattened before shifting, after which the original shape is restored.");
  }
};


inline bool NumpyRollShape(const nnvm::NodeAttrs& attrs,
                           mxnet::ShapeVector *in_attrs,
                           mxnet::ShapeVector *out_attrs) {
  using namespace mshadow;
  const NumpyRollParam& param = nnvm::get<NumpyRollParam>(attrs.parsed);

  if (!param.shift.has_value()) {
    LOG(FATAL) << "roll missing 1 required positional argument: 'shift'";
  }
  if (param.shift.value().ndim() > 1 &&
      param.axis.has_value() &&
      param.axis.value().ndim() != param.shift.value().ndim()) {
    LOG(FATAL) << "shift and `axis` must be a tuple of the same size,";
  }
  if (!param.axis.has_value() && param.shift.has_value() && param.shift.value().ndim() > 1) {
    LOG(FATAL) << "shift must be an int";
  }
  if (param.axis.has_value()) {
    mxnet::TShape axes(param.axis.value());
    const index_t ndim = (*in_attrs)[0].ndim();
    for (index_t i = 0; i < axes.ndim(); i++) {
      if (axes[i] < 0) {
        axes[i] += ndim;
      }
    }
    std::sort(axes.begin(), axes.end());
    for (index_t i = 1; i < axes.ndim(); i++) {
      CHECK_LT(axes[i - 1], axes[i])
        << "axes have duplicates " << axes;
    }
    CHECK_LT(axes[axes.ndim() - 1], ndim)
      << "axis " << axes[axes.ndim() - 1]
      << " Exceeds input dimensions " << (*in_attrs)[0];
    CHECK_GE(axes[0], 0)
      << "Reduction axis " << param.axis.value()
      << " Exceeds input dimensions " << (*in_attrs)[0];
  }
  return ElemwiseShape<1, 1>(attrs, in_attrs, out_attrs);
}

template<int req>
struct RollAxisNone_forward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* in_data,
                                  const int size, const int shift) {
    int new_index = i - shift < 0 ? i - shift + size : i - shift;
    KERNEL_ASSIGN(out_data[i], req, in_data[new_index]);
  }
};

template<int req>
struct RollAxis_forward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* in_data,
                                  const size_t* new_index) {
    KERNEL_ASSIGN(out_data[i], req, in_data[new_index[i]]);
  }
};

inline void dfs(std::vector<std::vector<size_t>>& new_axes,
                std::vector<size_t>& value,
                std::vector<size_t>& new_index,
                int index, int ndim, int mid) {
  for(int a : new_axes[index]) {
    if (index == ndim - 1) {
      new_index.push_back(mid + a);
    } else {
      mid += a * value[ndim - 1 - index];
      dfs(new_axes, value, new_index, index + 1, ndim, mid);
      mid -= a * value[ndim - 1 - index];
    }
  }
}


template<typename xpu>
void NumpyRollCompute(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  if (inputs[0].Size() == 0U) return;
  const NumpyRollParam& param = nnvm::get<NumpyRollParam>(attrs.parsed);
  const index_t ndim(inputs[0].shape_.ndim());
  Stream<xpu> *s = ctx.get_stream<xpu>();
  std::vector<int> shifts(ndim, 0);
  index_t input_size = inputs[0].Size();
  if (!param.axis.has_value()) {
    int shift = param.shift.value()[0];
    shift = shift % input_size;
    if (shift < 0) {
      shift += inputs[0].shape_.Size();
    }
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
          Kernel<RollAxisNone_forward<req_type>, xpu>::Launch(
              s, outputs[0].Size(), outputs[0].dptr<DType>(), inputs[0].dptr<DType>(),
              inputs[0].Size(), shift);
      });
    });
  } else {
    mxnet::TShape axes(param.axis.value());
    for (int i = 0; i < axes.ndim(); ++i) {
      if (axes[i] < 0)
        axes[i] += ndim;
    }
    for (int i = 0; i < axes.ndim(); ++i) {
      CHECK_LT(axes[i], ndim)
        << "axis " << axes[i]
        << " Exceeds input dimensions " << inputs[0].shape_;
      CHECK_GE(axes[0], 0)
        << "Reduction axis " << param.axis.value()
        << " Exceeds input dimensions " << inputs[0].shape_;
    }
    if (param.shift.value().ndim() == 1) {
      for (int i = 0; i < axes.ndim(); ++i) {
        shifts[axes[i]] = param.shift.value()[0];
      }
    } else {
      if (param.shift.value().ndim() != axes.ndim()) {
        LOG(FATAL) << "shift and `axis` must be a tuple of the same size,";
      }
      for (int i = 0; i < axes.ndim(); ++i) {
        shifts[axes[i]] = param.shift.value()[i];
      }
    }
    // keep shift in a legal range
    for (int i = 0; i < ndim; ++i) {
      int trans_shift = shifts[i] % inputs[0].shape_[i];
      if (trans_shift < 0) {
        trans_shift = shifts[i] + inputs[0].shape_[i];
      }
      shifts[i] = trans_shift;
    }

    // the result of new axis after shift.
    std::vector<std::vector<size_t>> new_axes;
    std::vector<size_t> new_index;
    std::vector<size_t> temp;
    std::vector<size_t> value(ndim, 0);
    int mid_val = 1;
    for (int i = 0; i < ndim; ++i) {
      if (shifts[i] != 0) {
        for (int j = 0; j < inputs[0].shape_[i]; ++j) {
          int new_axis = (j + inputs[0].shape_[i] - shifts[i]) % inputs[0].shape_[i];
          temp.push_back(new_axis);
        }
      } else {
        for (int j = 0; j < inputs[0].shape_[i]; ++j) {
          temp.push_back(j);
        }
      }
      new_axes.push_back(temp);
      temp.clear();
      value[i] = mid_val;
      mid_val *= inputs[0].shape_[ndim - 1 - i];
    }
    dfs(new_axes, value, new_index, 0, ndim, 0);
    size_t workspace_size = new_index.size() * sizeof(size_t);
    Tensor<xpu, 1, char> workspace =
        ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
    Tensor<cpu, 1, size_t> index_cpu_tensor(new_index.data(), Shape1(new_index.size()));
    Tensor<xpu, 1, size_t> index_xpu_tensor(
        reinterpret_cast<size_t*>(workspace.dptr_), Shape1(new_index.size()));
    mshadow::Copy(index_xpu_tensor, index_cpu_tensor, s);
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
          Kernel<RollAxis_forward<req_type>, xpu>::Launch(
              s, outputs[0].Size(), outputs[0].dptr<DType>(), inputs[0].dptr<DType>(),
              index_xpu_tensor.dptr_);
      });
    });
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_ROLL_OP_CC_H_
