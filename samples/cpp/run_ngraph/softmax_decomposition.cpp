/* ============================================================================
 * INTEL CONFIDENTIAL
 *
 * Copyright 2021 Intel Corporation All Rights Reserved.
 *
 * The source code contained or described herein and all documents related to
 * the source code ("Material") are owned by Intel Corporation or its suppliers
 * or licensors. Title to the Material remains with Intel Corporation or its
 * suppliers and licensors. The Material contains trade secrets and proprietary
 * and confidential information of Intel or its suppliers and licensors. The
 * Material is protected by worldwide copyright and trade secret laws and
 * treaty provisions. No part of the Material may be used, copied, reproduced,
 * modified, published, uploaded, posted, transmitted, distributed, or
 * disclosed in any way without Intel's prior express written permission.
 *
 * No license under any patent, copyright, trade secret or other intellectual
 * property right is granted to or conferred upon you by disclosure or delivery
 * of the Materials, either expressly, by implication, inducement, estoppel or
 * otherwise. Any license under such intellectual property rights must be
 * express and approved by Intel in writing.
 * ============================================================================
 */

#include "softmax_decomposition.hpp"

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

using namespace ngraph;
using namespace op;

NGRAPH_RTTI_DEFINITION(ngraph::pass::SoftmaxDecomposition, "SoftmaxDecomposition", 0);
bool ngraph::pass::SoftmaxDecomposition::run_on_function(std::shared_ptr<ngraph::Function> f) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto softmax_v1 = std::dynamic_pointer_cast<v1::Softmax>(node);
        auto softmax_v8 = std::dynamic_pointer_cast<v8::Softmax>(node);
        if ((nullptr == softmax_v1) && (nullptr == softmax_v8)) {
            continue;
        }

        const Output<Node>& parent = (nullptr != softmax_v1) ? softmax_v1->input_value(0) : softmax_v8->input_value(0);
        auto softmax_shape =
            (nullptr != softmax_v1) ? softmax_v1->get_output_shape(0) : softmax_v8->get_output_shape(0);
        auto axis = (nullptr != softmax_v1) ? softmax_v1->get_axis() : softmax_v8->get_axis();
        auto auto_broadcast = (nullptr != softmax_v1) ? softmax_v1->get_autob() : softmax_v8->get_autob();
        size_t N = 1, C = 1, H, W;
        if (softmax_shape.size() == 4) {
            N = softmax_shape[0];
            C = softmax_shape[1];
            H = softmax_shape[2];
            W = softmax_shape[3];
        } else if (softmax_shape.size() == 3) {
            C = softmax_shape[0];
            H = softmax_shape[1];
            W = softmax_shape[2];
        } else if (softmax_shape.size() == 2) {
            H = softmax_shape[0];
            W = softmax_shape[1];
        } else {
            continue;
        }

        if (N != 1) {
            continue;  // Batch case not yet implemented
        } else if (C != 1) {
            continue;  // Multi-channel case not yet implemented
        } else if (axis != softmax_shape.size() - 1) {
            continue;  // only support row softmax at this time
        }

        // simple binary kernels used only for copying data
        std::vector<float> copy_weights(8 * 8, 0.0f);
        for (size_t i = 0; i < 8; i++) {  // create identity kernels
            copy_weights[i * 8 + i] = 1.0f;
        }
        std::vector<float> neg_broadcast_weights(8 * 8 * W, 0.0f);
        for (size_t i = 0; i < 8; i++) {  // create broadcast kernels
            for (size_t j = 0; j < W; j++) {
                for (size_t k = 0; k < 8; k++) {
                    if (k == i) {
                        neg_broadcast_weights[i * W * 8 + j * 8 + k] = -1.0f;
                    }
                }
            }
        }
        auto copy_weights_const = op::Constant::create(ngraph::element::f32, Shape{8, 8, 1, 1}, copy_weights);
        auto neg_broadcast_weights_const =
            op::Constant::create(ngraph::element::f32, Shape{8 * W, 8, 1, 1}, neg_broadcast_weights);

        // Prepare to perform softmax sum in parts
        size_t num_parts = 1;
        while (W / num_parts > 768) {  // 768 is maximum GNA1/2 kernel size
            num_parts *= 2;
        }
        // Abort if W is not divisible by power of 2
        if ((W / num_parts) * num_parts != W) {
            continue;
        }
        std::vector<float> avg_weights(8 * W / num_parts, 1.0f / W);
        std::vector<float> avg_broadcast(8 * W * num_parts, 0.0f);
        std::vector<float> minus_log_W(H * W, -log((float)W));
        std::vector<float> minus_log_W_partial(8 * W, -log((float)W));
        for (size_t i = 0; i < W * num_parts; i++) {
            avg_broadcast[i * 8] = 1.0f;
        }
        auto avg_weights_const = op::Constant::create(ngraph::element::f32, Shape{8, W / num_parts, 1, 1}, avg_weights);
        auto avg_broadcast_const =
            op::Constant::create(ngraph::element::f32, Shape{W, 8 * num_parts, 1, 1}, avg_broadcast);
        auto minus_log_W_const = op::Constant::create(ngraph::element::f32, Shape{1, H * W}, minus_log_W);
        auto minus_log_W_partial_const =
            op::Constant::create(ngraph::element::f32, Shape{1, 8 * W}, minus_log_W_partial);

        auto parent_1d = std::make_shared<ngraph::opset1::Reshape>(
            parent,
            op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, H * W})->output(0),
            false);
        auto parent_2d = std::make_shared<ngraph::opset1::Reshape>(
            parent,
            op::Constant::create(ngraph::element::i64, Shape{2}, {H, W})->output(0),
            false);
        OutputVector upstream;
        upstream.push_back(parent_2d);

        // GNA transpose is limited to minor dimension of 8 and kernels must be multiple of 8
        // so split and pad if necessary to achieve the ideal height
        size_t H_new, H_pad = 8 - (H % 8);
        H_pad = (H_pad == 8) ? 0 : H_pad;
        H_new = H + H_pad;
        auto new_shape = softmax_shape;
        new_shape[softmax_shape.size() - 2] = H_new;
        if (H_pad > 0) {
            std::vector<float> padding(H_pad * W, 0.0f);
            auto padding_const = op::Constant::create(ngraph::element::f32, Shape{H_pad, W}, padding);
            OutputVector chunks;
            chunks.push_back(parent_2d);
            chunks.push_back(padding_const->output(0));
            auto new_concat = std::make_shared<ngraph::opset1::Concat>(chunks, 0);
            upstream[0] = new_concat->output(0);
        }

        size_t num_splits = H_new / 8;
        if (num_splits == 1) {
            auto transpose =
                std::make_shared<op::Transpose>(upstream[0],
                                                op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
            auto W_new = W;
            upstream[0] = transpose->output(0);
            // The fastest way to find the maximum is to use GNA pool size 6
            // and iteratively reduce to a single pool of size between 1 and 6
            while (W_new > 6) {
                size_t W_pad = 6 - (W_new % 6);
                W_pad = (W_pad == 6) ? 0 : W_pad;
                W_new = W_new + W_pad;
                if (W_pad > 0) {
                    std::vector<float> padding(8 * W_pad, BIG_NEGATIVE_NUMBER);
                    auto padding_const = op::Constant::create(ngraph::element::f32, Shape{W_pad, 8}, padding);
                    OutputVector chunks;
                    chunks.push_back(upstream[0]);
                    chunks.push_back(padding_const->output(0));
                    auto new_concat = std::make_shared<ngraph::opset1::Concat>(chunks, 0);
                    upstream[0] = new_concat->output(0);
                }
                auto reshape_2 = std::make_shared<ngraph::opset1::Reshape>(
                    upstream[0],
                    op::Constant::create(ngraph::element::i64, Shape{4}, {1ull, W_new, 1ull, 8ull})->output(0),
                    false);
                auto transpose_1 =
                    std::make_shared<op::Transpose>(reshape_2->output(0),
                                                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
                auto conv_1 = std::make_shared<opset1::Convolution>(transpose_1->output(0),
                                                                    copy_weights_const->output(0),
                                                                    Strides{1, 1},
                                                                    CoordinateDiff{0, 0},
                                                                    CoordinateDiff{0, 0},
                                                                    Strides{1, 1},
                                                                    PadType::VALID);
                auto pool_1 = std::make_shared<opset1::MaxPool>(conv_1->output(0),
                                                                Strides{6, 1},
                                                                Shape{0, 0},
                                                                Shape{0, 0},
                                                                Shape{6, 1},
                                                                ov::op::RoundingType::FLOOR,
                                                                op::PadType::VALID);
                auto transpose_2 =
                    std::make_shared<op::Transpose>(pool_1->output(0),
                                                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
                W_new = W_new / 6;
                auto reshape_3 = std::make_shared<ngraph::opset1::Reshape>(
                    transpose_2->output(0),
                    op::Constant::create(ngraph::element::i64, Shape{2}, {W_new, 8ull})->output(0),
                    false);
                upstream[0] = reshape_3->output(0);
            }
            // process the final pool to produce 1x8 vector of 8 maximums
            auto reshape_4 = std::make_shared<ngraph::opset1::Reshape>(
                upstream[0],
                op::Constant::create(ngraph::element::i64, Shape{4}, {1ull, W_new, 1ull, 8ull})->output(0),
                false);
            auto transpose_3 =
                std::make_shared<op::Transpose>(reshape_4->output(0),
                                                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
            auto conv_2 = std::make_shared<opset1::Convolution>(transpose_3->output(0),
                                                                copy_weights_const->output(0),
                                                                Strides{1, 1},
                                                                CoordinateDiff{0, 0},
                                                                CoordinateDiff{0, 0},
                                                                Strides{1, 1},
                                                                PadType::VALID);
            auto pool_2 = std::make_shared<opset1::MaxPool>(conv_2->output(0),
                                                            Strides{W_new, 1},
                                                            Shape{0, 0},
                                                            Shape{0, 0},
                                                            Shape{W_new, 1},
                                                            ov::op::RoundingType::FLOOR,
                                                            op::PadType::VALID);
            auto transpose_4 =
                std::make_shared<op::Transpose>(pool_2->output(0),
                                                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
            // broadcast negagtive of result back to original length
            auto reshape_5 = std::make_shared<ngraph::opset1::Reshape>(
                transpose_4->output(0),
                op::Constant::create(ngraph::element::i64, Shape{4}, {1ull, 1ull, 1ull, 8ull})->output(0),
                false);
            auto transpose_5 =
                std::make_shared<op::Transpose>(reshape_5->output(0),
                                                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
            auto conv_3 = std::make_shared<opset1::Convolution>(transpose_5->output(0),
                                                                neg_broadcast_weights_const->output(0),
                                                                Strides{1, 1},
                                                                CoordinateDiff{0, 0},
                                                                CoordinateDiff{0, 0},
                                                                Strides{1, 1},
                                                                PadType::VALID);
            auto transpose_6 =
                std::make_shared<op::Transpose>(conv_3->output(0),
                                                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
            auto reshape_6 = std::make_shared<op::v1::Reshape>(
                transpose_6->output(0),
                op::Constant::create(ngraph::element::i64, Shape{2}, {H_new, W})->output(0),
                false);
            // Remove padding
            auto slice_6 = std::make_shared<ngraph::opset1::StridedSlice>(
                reshape_6->output(0),
                ngraph::opset1::Constant::create(ngraph::element::i64,
                                                 ngraph::Shape{2},
                                                 {0ull, 0ull}),  // begin slice index
                ngraph::opset1::Constant::create(ngraph::element::i64,
                                                 ngraph::Shape{2},
                                                 {8 - (H_new - H), 0ull}),  // end slice index
                ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1ull, 1ull}),  // strides
                std::vector<int64_t>{0, 1},                                                              // begin mask
                std::vector<int64_t>{0, 1},                                                              // end mask
                std::vector<int64_t>{0, 0},
                std::vector<int64_t>{0, 0},
                std::vector<int64_t>{0, 0});
            auto reshape_6b = std::make_shared<op::v1::Reshape>(
                slice_6->output(0),
                op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, H * W})->output(0),
                false);
            // Subtract the maximum from each vector
            auto x_minus_max = std::make_shared<op::v1::Add>(parent_1d->output(0), reshape_6b->output(0));
            // perform softmax in log domain
            auto exp_x_minus_max = std::make_shared<op::Exp>(x_minus_max->output(0));
            auto reshape_7 = std::make_shared<op::v1::Reshape>(
                exp_x_minus_max->output(0),
                op::Constant::create(ngraph::element::i64, Shape{4}, {1ull, H * num_parts, 1ull, W / num_parts})
                    ->output(0),
                false);
            auto transpose_7 =
                std::make_shared<op::Transpose>(reshape_7->output(0),
                                                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
            auto transposed_avg_conv_1 = std::make_shared<opset1::Convolution>(transpose_7->output(0),
                                                                               avg_weights_const->output(0),
                                                                               Strides{1, 1},
                                                                               CoordinateDiff{0, 0},
                                                                               CoordinateDiff{0, 0},
                                                                               Strides{1, 1},
                                                                               PadType::VALID);
            auto avg_conv_1 =
                std::make_shared<op::Transpose>(transposed_avg_conv_1->output(0),
                                                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
            auto reshape_avg_conv_1 = std::make_shared<ngraph::opset1::Reshape>(
                avg_conv_1->output(0),
                op::Constant::create(ngraph::element::i64, Shape{4}, {N, 1ull, H, 8 * num_parts})->output(0),
                false);
            auto transpose_8 =
                std::make_shared<op::Transpose>(reshape_avg_conv_1->output(0),
                                                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
            auto transposed_avg_conv_2 = std::make_shared<opset1::Convolution>(transpose_8->output(0),
                                                                               avg_broadcast_const->output(0),
                                                                               Strides{1, 1},
                                                                               CoordinateDiff{0, 0},
                                                                               CoordinateDiff{0, 0},
                                                                               Strides{1, 1},
                                                                               PadType::VALID);
            auto avg_conv_2 =
                std::make_shared<op::Transpose>(transposed_avg_conv_2->output(0),
                                                op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
            auto avg_conv_2_1d = std::make_shared<ngraph::opset1::Reshape>(
                avg_conv_2,
                op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, H * W})->output(0),
                false);
            auto log_avg_1d = std::make_shared<op::Log>(avg_conv_2_1d->output(0));
            auto diff_1 = std::make_shared<op::v1::Add>(x_minus_max->output(0), minus_log_W_const->output(0));
            auto diff_2 = std::make_shared<op::v1::Subtract>(diff_1->output(0), log_avg_1d->output(0));
            auto softmax_output_1d = std::make_shared<op::Exp>(diff_2->output(0));
            if (softmax_shape.size() == 2) {
                auto reshape_8 = std::make_shared<ngraph::opset1::Reshape>(
                    softmax_output_1d->output(0),
                    op::Constant::create(ngraph::element::i64, Shape{2}, {H, W})->output(0),
                    false);
                if (nullptr != softmax_v1) {
                    ngraph::replace_node(softmax_v1, reshape_8);
                } else {
                    ngraph::replace_node(softmax_v8, reshape_8);
                }
            } else if (softmax_shape.size() == 3) {
                auto reshape_8 = std::make_shared<ngraph::opset1::Reshape>(
                    softmax_output_1d->output(0),
                    op::Constant::create(ngraph::element::i64, Shape{3}, {1ull, H, W})->output(0),
                    false);
                if (nullptr != softmax_v1) {
                    ngraph::replace_node(softmax_v1, reshape_8);
                } else {
                    ngraph::replace_node(softmax_v8, reshape_8);
                }
            } else {
                auto reshape_8 = std::make_shared<ngraph::opset1::Reshape>(
                    softmax_output_1d->output(0),
                    op::Constant::create(ngraph::element::i64, Shape{4}, {1ull, 1ull, H, W})->output(0),
                    false);
                if (nullptr != softmax_v1) {
                    ngraph::replace_node(softmax_v1, reshape_8);
                } else {
                    ngraph::replace_node(softmax_v8, reshape_8);
                }
            }
            is_graph_modfied = true;
        } else {
            const auto axis_node = ngraph::opset1::Constant::create(element::i64, Shape{}, {new_shape.size() - 2});
            const auto split = std::make_shared<ngraph::opset1::Split>(upstream[0], axis_node, num_splits);
            OutputVector chunks;
            for (size_t s = 0; s < num_splits; s++) {
                auto reshape_1_1d = std::make_shared<ngraph::opset1::Reshape>(
                    split->output(s),
                    op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, 8 * W})->output(0),
                    false);
                auto reshape_1_2d = std::make_shared<ngraph::opset1::Reshape>(
                    split->output(s),
                    op::Constant::create(ngraph::element::i64, Shape{2}, {8ull, W})->output(0),
                    false);
                auto transpose_part =
                    std::make_shared<op::Transpose>(reshape_1_2d->output(0),
                                                    op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                auto W_new = W;
                OutputVector section;
                section.push_back(transpose_part->output(0));
                // The fastest way to find the maximum is to use GNA pool size 6
                // and iteratively reduce to a single pool of size between 1 and 6
                while (W_new > 6) {
                    size_t W_pad = 6 - (W_new % 6);
                    W_pad = (W_pad == 6) ? 0 : W_pad;
                    W_new = W_new + W_pad;
                    if (W_pad > 0) {
                        std::vector<float> padding(8 * W_pad, BIG_NEGATIVE_NUMBER);
                        auto padding_const = op::Constant::create(ngraph::element::f32, Shape{W_pad, 8}, padding);
                        OutputVector chunks;
                        chunks.push_back(section[0]);
                        chunks.push_back(padding_const->output(0));
                        auto new_concat = std::make_shared<ngraph::opset1::Concat>(chunks, 0);
                        section[0] = new_concat->output(0);
                    }
                    auto reshape_2 = std::make_shared<ngraph::opset1::Reshape>(
                        section[0],
                        op::Constant::create(ngraph::element::i64, Shape{4}, {1ull, W_new, 1ull, 8ull})->output(0),
                        false);
                    auto transpose_1 = std::make_shared<op::Transpose>(
                        reshape_2->output(0),
                        op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
                    auto conv_1 = std::make_shared<opset1::Convolution>(transpose_1->output(0),
                                                                        copy_weights_const->output(0),
                                                                        Strides{1, 1},
                                                                        CoordinateDiff{0, 0},
                                                                        CoordinateDiff{0, 0},
                                                                        Strides{1, 1},
                                                                        PadType::VALID);
                    auto pool_1 = std::make_shared<opset1::MaxPool>(conv_1->output(0),
                                                                    Strides{6, 1},
                                                                    Shape{0, 0},
                                                                    Shape{0, 0},
                                                                    Shape{6, 1},
                                                                    ov::op::RoundingType::FLOOR,
                                                                    op::PadType::VALID);
                    auto transpose_2 = std::make_shared<op::Transpose>(
                        pool_1->output(0),
                        op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
                    W_new = W_new / 6;
                    auto reshape_3 = std::make_shared<ngraph::opset1::Reshape>(
                        transpose_2->output(0),
                        op::Constant::create(ngraph::element::i64, Shape{2}, {W_new, 8ull})->output(0),
                        false);
                    section[0] = reshape_3->output(0);
                }
                // process the final pool to produce 1x8 vector of 8 maximums
                auto reshape_4 = std::make_shared<ngraph::opset1::Reshape>(
                    section[0],
                    op::Constant::create(ngraph::element::i64, Shape{4}, {1ull, W_new, 1ull, 8ull})->output(0),
                    false);
                auto transpose_3 =
                    std::make_shared<op::Transpose>(reshape_4->output(0),
                                                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
                auto conv_2 = std::make_shared<opset1::Convolution>(transpose_3->output(0),
                                                                    copy_weights_const->output(0),
                                                                    Strides{1, 1},
                                                                    CoordinateDiff{0, 0},
                                                                    CoordinateDiff{0, 0},
                                                                    Strides{1, 1},
                                                                    PadType::VALID);
                auto pool_2 = std::make_shared<opset1::MaxPool>(conv_2->output(0),
                                                                Strides{W_new, 1},
                                                                Shape{0, 0},
                                                                Shape{0, 0},
                                                                Shape{W_new, 1},
                                                                ov::op::RoundingType::FLOOR,
                                                                op::PadType::VALID);
                auto transpose_4 =
                    std::make_shared<op::Transpose>(pool_2->output(0),
                                                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
                // broadcast partial result back to original length
                auto reshape_5 = std::make_shared<ngraph::opset1::Reshape>(
                    transpose_4->output(0),
                    op::Constant::create(ngraph::element::i64, Shape{4}, {1ull, 1ull, 1ull, 8ull})->output(0),
                    false);
                auto transpose_5 =
                    std::make_shared<op::Transpose>(reshape_5->output(0),
                                                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
                auto conv_3 = std::make_shared<opset1::Convolution>(transpose_5->output(0),
                                                                    neg_broadcast_weights_const->output(0),
                                                                    Strides{1, 1},
                                                                    CoordinateDiff{0, 0},
                                                                    CoordinateDiff{0, 0},
                                                                    Strides{1, 1},
                                                                    PadType::VALID);
                auto transpose_6 =
                    std::make_shared<op::Transpose>(conv_3->output(0),
                                                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
                auto reshape_6 = std::make_shared<op::v1::Reshape>(
                    transpose_6->output(0),
                    op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, 8 * W})->output(0),
                    false);
                // Subtract the maximum from each vector
                auto x_minus_max = std::make_shared<op::v1::Add>(reshape_1_1d->output(0), reshape_6->output(0));
                // perform softmax in log domain
                auto exp_x_minus_max = std::make_shared<op::Exp>(x_minus_max->output(0));
                auto reshape_7 = std::make_shared<op::v1::Reshape>(
                    exp_x_minus_max->output(0),
                    op::Constant::create(ngraph::element::i64, Shape{4}, {1ull, 8 * num_parts, 1ull, W / num_parts})
                        ->output(0),
                    false);
                auto transpose_7 =
                    std::make_shared<op::Transpose>(reshape_7->output(0),
                                                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
                auto transposed_avg_conv_1 = std::make_shared<opset1::Convolution>(transpose_7->output(0),
                                                                                   avg_weights_const->output(0),
                                                                                   Strides{1, 1},
                                                                                   CoordinateDiff{0, 0},
                                                                                   CoordinateDiff{0, 0},
                                                                                   Strides{1, 1},
                                                                                   PadType::VALID);
                auto avg_conv_1 =
                    std::make_shared<op::Transpose>(transposed_avg_conv_1->output(0),
                                                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
                auto reshape_avg_conv_1 = std::make_shared<ngraph::opset1::Reshape>(
                    avg_conv_1->output(0),
                    op::Constant::create(ngraph::element::i64, Shape{4}, {N, 1ull, 8ull, 8 * num_parts})->output(0),
                    false);
                auto transpose_8 =
                    std::make_shared<op::Transpose>(reshape_avg_conv_1->output(0),
                                                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));
                auto transposed_avg_conv_2 = std::make_shared<opset1::Convolution>(transpose_8->output(0),
                                                                                   avg_broadcast_const->output(0),
                                                                                   Strides{1, 1},
                                                                                   CoordinateDiff{0, 0},
                                                                                   CoordinateDiff{0, 0},
                                                                                   Strides{1, 1},
                                                                                   PadType::VALID);
                auto avg_conv_2 =
                    std::make_shared<op::Transpose>(transposed_avg_conv_2->output(0),
                                                    op::Constant::create(element::Type_t::i64, Shape{4}, {0, 2, 3, 1}));
                auto avg_conv_2_1d = std::make_shared<ngraph::opset1::Reshape>(
                    avg_conv_2,
                    op::Constant::create(ngraph::element::i64, Shape{2}, {1ull, 8 * W})->output(0),
                    false);
                auto log_avg_1d = std::make_shared<op::Log>(avg_conv_2_1d->output(0));
                auto diff_1 =
                    std::make_shared<op::v1::Add>(x_minus_max->output(0), minus_log_W_partial_const->output(0));
                auto diff_2 = std::make_shared<op::v1::Subtract>(diff_1->output(0), log_avg_1d->output(0));
                auto softmax_output_1d = std::make_shared<op::Exp>(diff_2->output(0));
                if (softmax_shape.size() == 2) {
                    auto reshape_8 = std::make_shared<ngraph::opset1::Reshape>(
                        softmax_output_1d->output(0),
                        op::Constant::create(ngraph::element::i64, Shape{2}, {8ull, W})->output(0),
                        false);
                    chunks.push_back(reshape_8->output(0));
                } else if (softmax_shape.size() == 3) {
                    auto reshape_8 = std::make_shared<ngraph::opset1::Reshape>(
                        softmax_output_1d->output(0),
                        op::Constant::create(ngraph::element::i64, Shape{3}, {1ull, 8ull, W})->output(0),
                        false);
                    chunks.push_back(reshape_8->output(0));
                } else {
                    auto reshape_8 = std::make_shared<ngraph::opset1::Reshape>(
                        softmax_output_1d->output(0),
                        op::Constant::create(ngraph::element::i64, Shape{4}, {1ull, 1ull, 8ull, W})->output(0),
                        false);
                    chunks.push_back(reshape_8->output(0));
                }
            }
            // remove padded part if necessary
            if (H_new != H) {
                if (softmax_shape.size() == 2) {
                    auto slice = std::make_shared<ngraph::opset1::StridedSlice>(
                        chunks[chunks.size() - 1],
                        ngraph::opset1::Constant::create(ngraph::element::i64,
                                                         ngraph::Shape{2},
                                                         {0ull, 0ull}),  // begin slice index
                        ngraph::opset1::Constant::create(ngraph::element::i64,
                                                         ngraph::Shape{2},
                                                         {8 - (H_new - H), 0ull}),  // end slice index
                        ngraph::opset1::Constant::create(ngraph::element::i64,
                                                         ngraph::Shape{2},
                                                         {1ull, 1ull}),  // strides
                        std::vector<int64_t>{0, 1},                      // begin mask
                        std::vector<int64_t>{0, 1},                      // end mask
                        std::vector<int64_t>{0, 0},
                        std::vector<int64_t>{0, 0},
                        std::vector<int64_t>{0, 0});
                    chunks[chunks.size() - 1] = slice->output(0);
                } else if (softmax_shape.size() == 3) {
                    auto slice = std::make_shared<ngraph::opset1::StridedSlice>(
                        chunks[chunks.size() - 1],
                        ngraph::opset1::Constant::create(ngraph::element::i64,
                                                         ngraph::Shape{3},
                                                         {0ull, 0ull, 0ull}),  // begin slice index
                        ngraph::opset1::Constant::create(ngraph::element::i64,
                                                         ngraph::Shape{3},
                                                         {0ull, 8 - (H_new - H), 0ull}),  // end slice index
                        ngraph::opset1::Constant::create(ngraph::element::i64,
                                                         ngraph::Shape{3},
                                                         {1ull, 1ull, 1ull}),  // strides
                        std::vector<int64_t>{1, 0, 1},                         // begin mask
                        std::vector<int64_t>{1, 0, 1},                         // end mask
                        std::vector<int64_t>{0, 0, 0},
                        std::vector<int64_t>{0, 0, 0},
                        std::vector<int64_t>{0, 0, 0});
                    chunks[chunks.size() - 1] = slice->output(0);
                } else {
                    auto slice = std::make_shared<ngraph::opset1::StridedSlice>(
                        chunks[chunks.size() - 1],
                        ngraph::opset1::Constant::create(ngraph::element::i64,
                                                         ngraph::Shape{4},
                                                         {0ull, 0ull, 0ull, 0ull}),  // begin slice index
                        ngraph::opset1::Constant::create(ngraph::element::i64,
                                                         ngraph::Shape{4},
                                                         {0ull, 0ull, 8 - (H_new - H), 0ull}),  // end slice index
                        ngraph::opset1::Constant::create(ngraph::element::i64,
                                                         ngraph::Shape{4},
                                                         {1ull, 1ull, 1ull, 1ull}),  // strides
                        std::vector<int64_t>{1, 1, 0, 1},                            // begin mask
                        std::vector<int64_t>{1, 1, 0, 1},                            // end mask
                        std::vector<int64_t>{0, 0, 0, 0},
                        std::vector<int64_t>{0, 0, 0, 0},
                        std::vector<int64_t>{0, 0, 0, 0});
                    chunks[chunks.size() - 1] = slice->output(0);
                }
            }
            auto concat = std::make_shared<ngraph::opset1::Concat>(chunks, softmax_shape.size() - 2);
            if (nullptr != softmax_v1) {
                ngraph::replace_node(softmax_v1, concat);
            } else {
                ngraph::replace_node(softmax_v8, concat);
            }
            is_graph_modfied = true;
        }
    }
    return is_graph_modfied;
}
