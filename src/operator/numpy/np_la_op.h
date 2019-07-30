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
 * \file np_la_op.h
 * \brief Function definition of Operators for advanced linear algebra.
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_LA_OP_H_
#define MXNET_OPERATOR_NUMPY_NP_LA_OP_H_

#include <tuple>
#include "../tensor/la_op.h"
#include "../tensor/broadcast_reduce_op.h"
#include "../tensor/broadcast_reduce-inl.h"
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {
using namespace mshadow;

struct NumpyLaNorm : public dmlc::Parameter<NumpyLaNorm> {
  int ord;
  dmlc::optional<mxnet::TShape> axis;
  bool keepdims;
  int flag;
  DMLC_DECLARE_PARAMETER(NumpyLaNorm) {
      DMLC_DECLARE_FIELD(ord).set_default(2)
          .describe("Order of the norm. inf means numpy’s inf object.");
      DMLC_DECLARE_FIELD(axis).set_default(dmlc::optional<mxnet::TShape>())
      .describe(R"code(If axis is an integer, it specifies the axis of x along
         which to compute the vector norms. If axis is a 2-tuple,
         it specifies the axes that hold 2-D matrices, and the matrix norms of
         these matrices are computed. If axis is None then either a vector norm (when x is 1-D)
         or a matrix norm (when x is 2-D) is returned.If axis is an integer,
         it specifies the axis of x along which to compute the vector norms.
         If axis is a 2-tuple, it specifies the axes that hold 2-D matrices,
         and the matrix norms of these matrices are computed. If axis is None then either a
         vector norm (when x is 1-D) or a matrix norm (when x is 2-D) is returned.)code");
      DMLC_DECLARE_FIELD(keepdims).set_default(false)
      .describe("If this is set to `True`, the reduced axis is left "
      "in the result as dimension with size one.");
      DMLC_DECLARE_FIELD(flag).set_default(-1)
      .describe("{ord: None  flag:0, ord: 'fro'  flag:1, ord: 'nuc'  flag:2"
      " ord: inf  flag:3  ord: -inf  flag:4 }");
  }
};

inline mxnet::TShape NumpyLaNormShapeImpl(const mxnet::TShape& ishape,
                                          const int ord,
                                          const dmlc::optional<mxnet::TShape>& axis,
                                          const int flag,
                                          bool keepdims, bool exclude) {

  const int ndim = ishape.ndim();

  if ((!axis.has_value() && flag != 0 && ndim > 2) || (axis.has_value() && axis.value().ndim() > 2))
    LOG(FATAL) << "Improper number of dimensions to norm.";

  if (!axis.has_value()) {
    if ((ndim == 0 && flag != 0) ||    // for scalar
        (ndim == 1 && (flag == 1 || flag ==2)) ||
        (ndim >= 2 && (ord == 0 || ord > 2 || ord < -2))) {
      LOG(FATAL) << "Invalid norm order for inputs.";
    }
  } else {
    if ((axis.value().ndim() == 0 && flag != 0) ||    // for scalar
        (axis.value().ndim()  == 1 && (flag == 1 || flag ==2)) ||
        (axis.value().ndim()  == 2 && (ord == 0 || ord > 2 || ord < -2))) {
      LOG(FATAL) << "Invalid norm order for inputs.";
    }
  }

  if (axis.has_value() && axis.value().ndim() == 0) {
    if (keepdims) {
      return mxnet::TShape(ishape.ndim(), 1);
    } else {
      return mxnet::TShape(1, 1);
    }
  }

  bool flag_axis = false;
  if (!axis.has_value()) {
    flag_axis = false;
  } else {
    flag_axis = true;
  }
  mxnet::TShape axes(flag_axis ? axis.value().ndim() : ndim, 0);
  // axis has value
  if (flag_axis) {
    axes = axis.value();
  } else {
    for (int i = 0; i < ndim; ++i) {
      axes[i] = i;
    }
  }
  for (index_t i = 0; i < axes.ndim(); i++) {
    if (axes[i] < 0) {
      axes[i] += ishape.ndim();
    }
  }
  std::sort(axes.begin(), axes.end());

  for (index_t i = 1; i < axes.ndim(); i++) {
    CHECK_LT(axes[i-1], axes[i])
        << "Reduction axes have duplicates "
        << axes;
  }
  CHECK_LT(axes[axes.ndim()-1], ishape.ndim())
      << "Reduction axis " << axes[axes.ndim()-1]
      << " Exceeds input dimensions " << ishape;
  CHECK_GE(axes[0], 0)
      << "Reduction axis " << axis.value()
      << " Exceeds input dimensions " << ishape;

  if (axes.ndim() == 2) {
    int row_axis = axes[0], col_axis = axes[1];

    if (row_axis == col_axis)
      LOG(FATAL) << "Duplicate axes given.";
    else if (flag == 2 || (flag == -1 && ord ==2) || ord == -2) {
      LOG(FATAL) << "Not implement svd for norm.";
    }
  }

  mxnet::TShape oshape;
  if (keepdims) {
    oshape = mxnet::TShape(ishape);
  } else if (exclude) {
    oshape = mxnet::TShape(axes.ndim(), 1);
  } else {
    oshape = mxnet::TShape(std::max(1, ishape.ndim() - axes.ndim()), 1);
  }

  if (keepdims && exclude) {
    for (index_t i = 0, j = 0; i < ishape.ndim(); ++i) {
      if (j < axes.ndim() && i == axes[j]) {
        ++j;
        continue;
      }
      oshape[i] = 1;
    }
  } else if (keepdims) {
    for (index_t i = 0; i < axes.ndim(); ++i) {
      oshape[axes[i]] = 1;
    }
  } else if (exclude) {
    for (index_t i = 0; i < axes.ndim(); ++i) {
      oshape[i] = ishape[axes[i]];
    }
  } else {
    for (index_t i = 0, j = 0, k = 0; i < ishape.ndim(); ++i) {
      if (j < axes.ndim() && i == axes[j]) {
        ++j;
        continue;
      }
      oshape[k++] = ishape[i];
    }
  }
  return oshape;
}



inline mxnet::TShape NumpyScalarLaNormShapeImpl(const mxnet::TShape& ishape,
//                                          const int ord,
                                          const int axis,
//                                          const int flag,
                                          bool keepdims, bool exclude) {

  const int ndim = ishape.ndim();
  mxnet::TShape axes(1, axis);
  mxnet::TShape oshape;
  if (keepdims) {
    oshape = mxnet::TShape(ishape);
  } else if (exclude) {
    oshape = mxnet::TShape(axes.ndim(), 1);
  } else {
    oshape = mxnet::TShape(std::max(1, ishape.ndim() - axes.ndim()), 1);
  }

  if (keepdims && exclude) {
    for (index_t i = 0, j = 0; i < ishape.ndim(); ++i) {
      if (j < axes.ndim() && i == axes[j]) {
        ++j;
        continue;
      }
      oshape[i] = 1;
    }
  } else if (keepdims) {
    for (index_t i = 0; i < axes.ndim(); ++i) {
      oshape[axes[i]] = 1;
    }
  } else if (exclude) {
    for (index_t i = 0; i < axes.ndim(); ++i) {
      oshape[i] = ishape[axes[i]];
    }
  } else {
    for (index_t i = 0, j = 0, k = 0; i < ishape.ndim(); ++i) {
      if (j < axes.ndim() && i == axes[j]) {
        ++j;
        continue;
      }
      oshape[k++] = ishape[i];
    }
  }
  return oshape;
}

inline bool NumpyLaNormShape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector *in_attrs,
                             mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  if (!shape_is_known((*in_attrs)[0])) return false;
  const NumpyLaNorm& param = nnvm::get<NumpyLaNorm>(attrs.parsed);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0,
                     NumpyLaNormShapeImpl((*in_attrs)[0], param.ord, param.axis,
                                          param.flag, param.keepdims, false));
  return true;
}

template<int req, typename OP>
struct numpy_norm_reduce_axes_backward_broadcast{
  template<typename DType, typename OType>
  MSHADOW_XINLINE static void Map(index_t i,
                                  DType *data,
                                  OType *out,
                                  DType *igrad,
                                  OType *ograd,
                                  mshadow::Shape<5> in_shape,
                                  mshadow::Shape<5> out_shape,
                                  const uint32_t ndim,
                                  const float ord) {
    size_t in_stride = 1;
    size_t out_stride = 1;
    index_t idx = i;
    index_t out_idx = i;
    for (int iter = ndim - 1; iter >= 0; --iter) {
      size_t dim_idx = idx % in_shape[iter];
      out_idx -= dim_idx * in_stride;
      if (out_shape[iter] != 1) {
        out_idx += dim_idx * out_stride;
      }
      idx /= in_shape[iter];
      in_stride *= in_shape[iter];
      out_stride *= out_shape[iter];
    }
    int flag = mshadow_op::sign::Map(data[i]);
    KERNEL_ASSIGN(igrad[i], req, DType(ograd[out_idx]) * OP::Map(DType(mshadow_op::abs(data[i])), DType(out[out_idx]))* DType(flag));
  }
};

template <int req, typename OP>
struct numpy_norm_power_backward_broadcast {
  template<typename DType, typename OType>
  MSHADOW_XINLINE static void Map(index_t i,
                                  DType *data,
                                  OType *out,
                                  DType *igrad,
                                  OType *ograd,
                                  mshadow::Shape<5> in_shape,
                                  mshadow::Shape<5> out_shape,
                                  const uint32_t ndim,
                                  const float ord) {
    size_t in_stride = 1;
    size_t out_stride = 1;
    index_t idx = i;
    index_t out_idx = i;
    for (int iter = ndim - 1; iter >= 0; --iter) {
      size_t dim_idx = idx % in_shape[iter];
      out_idx -= dim_idx * in_stride;
      if (out_shape[iter] != 1) {
        out_idx += dim_idx * out_stride;
      }
      idx /= in_shape[iter];
      in_stride *= in_shape[iter];
      out_stride *= out_shape[iter];
    }
    int flag = mshadow_op::sign::Map(data[i]);
    DType temp = mshadow_op::abs(data[i]);
    KERNEL_ASSIGN(igrad[i], req, DType(ograd[out_idx]) * DType(out[out_idx])*
                  OP::Map(DType(math::pow(temp, ord-1)), DType(math::pow(out[out_idx], ord))));
  }
};


/*! \brief compute ln norm */
struct nrmn {
  /*! \brief do reduction into dst */
  template<typename AType, typename DType>
  MSHADOW_XINLINE static void Reduce(volatile AType& sum_of_power,  volatile DType src, volatile DType& scale) { // NOLINT(*)
    DType abs = mshadow_op::abs::Map(src);
    sum_of_power += math::pow(src, scale);
  }

  /*! \brief finalize reduction result */
  template<typename DType>
  MSHADOW_XINLINE static void Finalize(volatile DType& sum_of_power, volatile DType& scale) { // NOLINT(*)
    sum_of_power = math::pow(sum_of_power, 1/scale);
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &sum_of_power) { // NOLINT(*)
    sum_of_power = 0;
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &sum_of_power, DType &scale) { // NOLINT(*)
    SetInitValue(sum_of_power);
    scale = SetInitValue;
  }
};



template<typename Reducer, int ndim, typename AType, typename DType, typename OType, typename OP>
void norm_seq_reduce_compute(const size_t N, const size_t M, const bool addto,
                        const DType *big, OType *small, const Shape<ndim> bshape,
                        const Shape<ndim> sshape, const Shape<ndim> rshape,
                        const Shape<ndim> rstride, const float ord) {

  #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (index_t idx = 0; idx < static_cast<index_t>(N); ++idx) {
    Shape<ndim> coord = unravel(idx, sshape);
    index_t j = ravel(coord, bshape);
    AType val, residual;
    residual = AType(ord);
    Reducer::SetInitValue(val, residual);
    for (size_t k = 0; k < M; ++k) {
      coord = unravel(k, rshape);
      Reducer::Reduce(val, AType(OP::Map(big[j + dot(coord, rstride)])), residual);
    }
    Reducer::Finalize(val, residual);
    assign(&small[idx], addto, OType(val));
  }
}

template <typename Reducer, int ndim, typename DType, typename OP, bool safe_acc = false>
void PowerReduce(Stream<cpu>* s, const TBlob& small, const OpReqType req,
            const Tensor<cpu, 1, char>& workspace, const TBlob& big, const float ord) {
  if (req == kNullOp) return;
  Shape<ndim> rshape, rstride;
  diff(small.shape_.get<ndim>(), big.shape_.get<ndim>(), &rshape, &rstride);
  size_t N = small.shape_.Size(), M = rshape.Size();
  if (!safe_acc) {
    norm_seq_reduce_compute<Reducer, ndim, DType, DType, DType, OP>(
        N, M, req == kAddTo, big.dptr<DType>(), small.dptr<DType>(),
        big.shape_.get<ndim>(), small.shape_.get<ndim>(), rshape, rstride, ord);
  } else {
    // TODO(haojin2): Use real-only type swtich for windows temporarily due to CI issues.
#ifndef _WIN32
    MXNET_ACC_TYPE_SWITCH(mshadow::DataType<DType>::kFlag, DataType, AType, {
      typedef typename std::conditional<safe_acc, AType, DataType>::type AccType;
      MSHADOW_TYPE_SWITCH(small.type_flag_, OType, {
          typedef typename std::conditional<safe_acc, OType, DataType>::type OutType;
          norm_seq_reduce_compute<Reducer, ndim, AccType, DataType, OutType, OP>(
          N, M, req == kAddTo, big.dptr<DataType>(), small.dptr<OutType>(),
          big.shape_.get<ndim>(), small.shape_.get<ndim>(), rshape, rstride, ord);
      });
    });
#else
    MXNET_REAL_ACC_TYPE_SWITCH(mshadow::DataType<DType>::kFlag, DataType, AType, {
      typedef typename std::conditional<safe_acc, AType, DataType>::type AccType;
      MSHADOW_TYPE_SWITCH(small.type_flag_, OType, {
        typedef typename std::conditional<safe_acc, OType, DataType>::type OutType;
        norm_seq_reduce_compute<Reducer, ndim, AccType, DataType, OutType, OP>(
          N, M, req == kAddTo, big.dptr<DataType>(), small.dptr<OutType>(),
          big.shape_.get<ndim>(), small.shape_.get<ndim>(), rshape, rstride, ord);
      });
    });
#endif
  }
}


template<typename xpu, typename reducer, bool safe_acc, bool normalize = false,
    typename OP = op::mshadow_op::identity>
void NumpyNormPowerComputeImpl(const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs,
                           const mxnet::TShape& small,
                           const float ord) {
  using namespace mshadow;
  using namespace mshadow::expr;

  mxnet::TShape src_shape, dst_shape;
  BroadcastReduceShapeCompact(inputs[0].shape_, small, &src_shape, &dst_shape);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
        const TBlob in_data = inputs[0].reshape(src_shape);
        const TBlob out_data = outputs[0].reshape(dst_shape);
        BROADCAST_NDIM_SWITCH(dst_shape.ndim(), NDim, {
          size_t workspace_size = broadcast::ReduceWorkspaceSize<NDim, DType>(
              s, out_data.shape_, req[0], in_data.shape_);
          Tensor<xpu, 1, char> workspace =
              ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
          PowerReduce<reducer, NDim, DType, OP, safe_acc>(
              s, out_data, req[0], workspace, in_data, ord);
          if (normalize) {
            auto out = out_data.FlatTo2D<xpu, OType>(s);
            out /= scalar<OType>(src_shape.Size()/dst_shape.Size());
          }
        });
    });
  });
}

template<typename xpu, typename OP, bool normalize = false>
void NumpyNormReduceAxesBackwardUseInOutImpl(const OpContext& ctx,
                                             const mxnet::TShape &small,
                                             const std::vector<TBlob>& inputs,
                                             const std::vector<OpReqType>& req,
                                             const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;

  mxnet::TShape src_shape, dst_shape;
  BroadcastReduceShapeCompact(outputs[0].shape_, small, &src_shape, &dst_shape);
  Stream<xpu> *s = ctx.get_stream<xpu>();

  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
      MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
          mshadow::Shape<5> in_shape;
          mshadow::Shape<5> out_shape;
          for (int i = 0; i < 5; ++i) {
            if (i < dst_shape.ndim()) {
              in_shape[i] = src_shape[i];
              out_shape[i] = dst_shape[i];
            } else {
              in_shape[i] = 1;
              out_shape[i] = 1;
            }
          }
          if (dst_shape.ndim() == 2) {
            Tensor<xpu, 2, OType> igrad =
                outputs[0].get_with_shape<xpu, 2, OType>(src_shape.get<2>(), s);
            Tensor<xpu, 2, DType> ograd =
                inputs[0].get_with_shape<xpu, 2, DType>(dst_shape.get<2>(), s);
            Tensor<xpu, 2, OType> data =
                inputs[1].get_with_shape<xpu, 2, OType>(src_shape.get<2>(), s);
            Tensor<xpu, 2, DType> out =
                inputs[2].get_with_shape<xpu, 2, DType>(dst_shape.get<2>(), s);
            MXNET_REQ_TYPE_SWITCH(req[0], Req, {
              Kernel<numpy_norm_reduce_axes_backward_broadcast<Req, OP>, xpu>::Launch(
                  s, outputs[0].shape_.Size(), data.dptr_, out.dptr_, igrad.dptr_, ograd.dptr_,
                  in_shape, out_shape, src_shape.ndim());
            });
            if (normalize) igrad /= scalar<OType>(src_shape.Size()/dst_shape.Size());
          } else {
            const int ndim = MXNET_SPECIAL_MAX_NDIM;
            Tensor<xpu, ndim, OType> igrad =
                outputs[0].get_with_shape<xpu, ndim, OType>(src_shape.get<ndim>(), s);
            Tensor<xpu, ndim, DType> ograd =
                inputs[0].get_with_shape<xpu, ndim, DType>(dst_shape.get<ndim>(), s);
            Tensor<xpu, ndim, OType> data =
                inputs[1].get_with_shape<xpu, ndim, OType>(src_shape.get<ndim>(), s);
            Tensor<xpu, ndim, DType> out =
                inputs[2].get_with_shape<xpu, ndim, DType>(dst_shape.get<ndim>(), s);
            MXNET_REQ_TYPE_SWITCH(req[0], Req, {
              Kernel<numpy_norm_reduce_axes_backward_broadcast<Req, OP>, xpu>::Launch(
                  s, outputs[0].shape_.Size(), data.dptr_, out.dptr_, igrad.dptr_, ograd.dptr_,
                  in_shape, out_shape, src_shape.ndim());
            });
            if (normalize) igrad /= scalar<OType>(src_shape.Size()/dst_shape.Size());
          }
      });
  });
}


template<typename xpu, typename OP, bool normalize = false>
void NumpyNormPowerBackwardUseInOutImpl(const OpContext& ctx,
                                             const mxnet::TShape &small,
                                             const std::vector<TBlob>& inputs,
                                             const std::vector<OpReqType>& req,
                                             const std::vector<TBlob>& outputs,
                                             const float ord) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;

  mxnet::TShape src_shape, dst_shape;
  BroadcastReduceShapeCompact(outputs[0].shape_, small, &src_shape, &dst_shape);
  Stream<xpu> *s = ctx.get_stream<xpu>();

  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
        mshadow::Shape<5> in_shape;
        mshadow::Shape<5> out_shape;
        for (int i = 0; i < 5; ++i) {
          if (i < dst_shape.ndim()) {
            in_shape[i] = src_shape[i];
            out_shape[i] = dst_shape[i];
          } else {
            in_shape[i] = 1;
            out_shape[i] = 1;
          }
        }
        if (dst_shape.ndim() == 2) {
          Tensor<xpu, 2, OType> igrad =
              outputs[0].get_with_shape<xpu, 2, OType>(src_shape.get<2>(), s);
          Tensor<xpu, 2, DType> ograd =
              inputs[0].get_with_shape<xpu, 2, DType>(dst_shape.get<2>(), s);
          Tensor<xpu, 2, OType> data =
              inputs[1].get_with_shape<xpu, 2, OType>(src_shape.get<2>(), s);
          Tensor<xpu, 2, DType> out =
              inputs[2].get_with_shape<xpu, 2, DType>(dst_shape.get<2>(), s);
          MXNET_REQ_TYPE_SWITCH(req[0], Req, {
              Kernel<numpy_norm_power_backward_broadcast<Req, OP>, xpu>::Launch(
                  s, outputs[0].shape_.Size(), data.dptr_, out.dptr_, igrad.dptr_, ograd.dptr_,
                  in_shape, out_shape, src_shape.ndim(), ord);
          });
          if (normalize) igrad /= scalar<OType>(src_shape.Size()/dst_shape.Size());
        } else {
          const int ndim = MXNET_SPECIAL_MAX_NDIM;
          Tensor<xpu, ndim, OType> igrad =
              outputs[0].get_with_shape<xpu, ndim, OType>(src_shape.get<ndim>(), s);
          Tensor<xpu, ndim, DType> ograd =
              inputs[0].get_with_shape<xpu, ndim, DType>(dst_shape.get<ndim>(), s);
          Tensor<xpu, ndim, OType> data =
              inputs[1].get_with_shape<xpu, ndim, OType>(src_shape.get<ndim>(), s);
          Tensor<xpu, ndim, DType> out =
              inputs[2].get_with_shape<xpu, ndim, DType>(dst_shape.get<ndim>(), s);
          MXNET_REQ_TYPE_SWITCH(req[0], Req, {
              Kernel<numpy_norm_power_backward_broadcast<Req, OP>, xpu>::Launch(
                  s, outputs[0].shape_.Size(), data.dptr_, out.dptr_, igrad.dptr_, ograd.dptr_,
                  in_shape, out_shape, src_shape.ndim(), ord);
          });
          if (normalize) igrad /= scalar<OType>(src_shape.Size()/dst_shape.Size());
        }
    });
  });
}


template<typename xpu>
void NumpyLaNormCompute(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace mshadow::expr;

  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const NumpyLaNorm& param = nnvm::get<NumpyLaNorm>(attrs.parsed);
  const int ndim = inputs[0].ndim();
  if (req[0] == kNullOp) return;
  mxnet::TShape small;
  mxnet::TShape mid;
  bool flag_axis = false;
  if (!param.axis.has_value()) {
    flag_axis = false;
  } else {
    flag_axis = true;
  }
  mxnet::TShape axes(flag_axis ? param.axis.value().ndim() : ndim, 0);
  // axis has value
  if (flag_axis) {
    axes = param.axis.value();
  } else {
    for (int i = 0; i < ndim; ++i) {
      axes[i] = i;
    }
  }

  if (param.keepdims) {
    small = outputs[0].shape_;
  } else {
    small = NumpyLaNormShapeImpl(inputs[0].shape_, param.ord, param.axis,
                                 param.flag, true, false);
  }
  bool safe_acc = dmlc::GetEnv("MXNET_SAFE_ACCUMULATION", false);
  if (!safe_acc && inputs[0].type_flag_ == mshadow::kFloat16) {
    common::LogOnce("MXNET_SAFE_ACCUMULATION=1 is recommended for LpNorm with float16 inputs. "
                    "See https://mxnet.incubator.apache.org/versions/master/faq/env_var.html "
                    "for more details.");
  }
  // Immediately handle some default, simple, fast, and common cases.
  if (!param.axis.has_value()) {
    if (param.flag == 0 || (param.flag == 1 && ndim == 2) || (param.flag == -1 && param.ord == 2 && ndim == 1)) {
      if (safe_acc) {
        ReduceAxesComputeImpl<xpu, mshadow_op::nrm2, true, false, mshadow_op::identity>(
            ctx, inputs, req, outputs, small);
      } else {
        ReduceAxesComputeImpl<xpu, mshadow_op::nrm2, false, false, mshadow_op::identity>(
            ctx, inputs, req, outputs, small);
      }
    }
  } else if (axes.ndim() == 1) {
    // if numpy ord is inf
    if (param.flag == 3) {
      if (safe_acc) {
        ReduceAxesComputeImpl<xpu, mshadow::red::maximum, true, false, mshadow_op::abs>(
            ctx, inputs, req, outputs, small);
      } else {
        ReduceAxesComputeImpl<xpu, mshadow::red::maximum, false, false, mshadow_op::abs>(
            ctx, inputs, req, outputs, small);
      }
    } else if (param.flag == 4) {
      if (safe_acc) {
        ReduceAxesComputeImpl<xpu, mshadow::red::minimum, true, false, mshadow_op::abs>(
            ctx, inputs, req, outputs, small);
      } else {
        ReduceAxesComputeImpl<xpu, mshadow::red::minimum, false, false, mshadow_op::abs>(
            ctx, inputs, req, outputs, small);
      }
    } else if (param.ord == 1) {
      if (safe_acc) {
        ReduceAxesComputeImpl<xpu, mshadow_op::sum, true, false, mshadow_op::abs>(
            ctx, inputs, req, outputs, small);
      } else {
        ReduceAxesComputeImpl<xpu, mshadow_op::sum, false, false, mshadow_op::abs>(
            ctx, inputs, req, outputs, small);
      }
    } else if (param.flag == 0 || (param.flag == -1 && param.ord == 2)) {
      if (safe_acc) {
        ReduceAxesComputeImpl<xpu, mshadow_op::nrm2, true, false, mshadow_op::identity>(
            ctx, inputs, req, outputs, small);
      } else {
        ReduceAxesComputeImpl<xpu, mshadow_op::nrm2, false, false, mshadow_op::identity>(
            ctx, inputs, req, outputs, small);
      }
    } else {
      if (safe_acc) {
        NumpyNormPowerComputeImpl<xpu, nrmn, true, false, mshadow_op::identity>(
            ctx, inputs, req, outputs, small, param.ord);
      } else {
        NumpyNormPowerComputeImpl<xpu, nrmn, false, false, mshadow_op::identity>(
            ctx, inputs, req, outputs, small, param.ord);
      }
    }
  } else if (axes.ndim() == 2) {
    if (param.flag == 2 || (param.flag == -1 && (param.ord == 2 || param.ord == -2))) {
      LOG(FATAL) << "Not implement svd.";
    } else if (param.flag == 0 || param.flag == 1) {
      if (safe_acc) {
        ReduceAxesComputeImpl<xpu, mshadow_op::nrm2, true, false, mshadow_op::identity>(
            ctx, inputs, req, outputs, small);
      } else {
        ReduceAxesComputeImpl<xpu, mshadow_op::nrm2, false, false, mshadow_op::identity>(
            ctx, inputs, req, outputs, small);
      }
    } else if (param.flag == 3 || param.flag == 4 || param.ord == 1 || param.ord == -1) {
      LOG(FATAL) << "Not implement in c++ backend";
    } else {
      LOG(FATAL) << "Invalid norm order for matrices.";
    }
  } else {
    LOG(FATAL) << "Improper number of dimensions to norm.";
  }
}


template<typename xpu>
void NumpyLpNormGradCompute(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  if (req[0] == kNullOp) return;

  const NumpyLaNorm& param = nnvm::get<NumpyLaNorm>(attrs.parsed);
  mxnet::TShape small;
  mxnet::TShape mid;
  const int ndim = inputs[0].ndim();
  bool flag_axis = false;
  if (!param.axis.has_value()) {
    flag_axis = false;
  } else {
    flag_axis = true;
  }
  mxnet::TShape axes(flag_axis ? param.axis.value().ndim() : ndim, 0);
  // axis has value
  if (flag_axis) {
    axes = param.axis.value();
  } else {
    for (int i = 0; i < ndim; ++i) {
      axes[i] = i;
    }
  }

  if (param.keepdims) {
    small = outputs[0].shape_;
  } else {
    small = NumpyLaNormShapeImpl(inputs[0].shape_, param.ord, param.axis,
                                 param.flag, true, false);
  }

  bool safe_acc = dmlc::GetEnv("MXNET_SAFE_ACCUMULATION", false);
  if (!safe_acc && inputs[0].type_flag_ == mshadow::kFloat16) {
    common::LogOnce("MXNET_SAFE_ACCUMULATION=1 is recommended for LpNorm with float16 inputs. "
                    "See https://mxnet.incubator.apache.org/versions/master/faq/env_var.html "
                    "for more details.");
  }

  // Immediately handle some default, simple, fast, and common cases.
  if (!param.axis.has_value()) {
    if (param.flag == 0 || (param.flag == 1 && ndim == 2) || (param.flag == -1 && param.ord == 2 && ndim == 1)) {
      ReduceAxesBackwardUseInOutImpl<xpu, mshadow_op::div, false>(ctx, small, inputs,
                                                                  req, outputs);
    }
  } else if (axes.ndim() == 1) {
    // if numpy ord is inf
    if (param.flag == 3 || param.flag == 4 || param.ord == 1) {
      NumpyNormReduceAxesBackwardUseInOutImpl<xpu, mshadow_op::eq, false>
            (ctx, small, inputs, req, outputs);
    } else if (param.flag == 0 || (param.flag == -1 && param.ord == 2)) {
      ReduceAxesBackwardUseInOutImpl<xpu, mshadow_op::div, false>(ctx, small, inputs,
                                                                  req, outputs);
    } else {
      NumpyNormPowerBackwardUseInOutImpl<xpu, mshadow_op::div, false>(ctx, small, inputs,
                                                                      req, outputs, param.ord);
    }
  } else if (axes.ndim() == 2) {
    if (param.flag == 2 || (param.flag == -1 && (param.ord == 2 || param.ord == -2))) {
      LOG(FATAL) << "Not implement svd.";
    } else if (param.flag == 0 || param.flag == 1) {
      ReduceAxesBackwardUseInOutImpl<xpu, mshadow_op::div, false>(ctx, small, inputs,
                                                                  req, outputs);
    } else if (param.flag == 3 || param.flag == 4 ||param.ord == 1 || param.ord == -1) {
      LOG(FATAL) << "Not implement in c++ backend";
     } else {
      LOG(FATAL) << "Invalid norm order for matrices.";
    }
  } else {
    LOG(FATAL) << "Improper number of dimensions to norm.";
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_LA_OP_H_
