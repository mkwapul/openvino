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

#include "matmul_decomposition.hpp"

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

using namespace ngraph;
using namespace op;

bool ngraph::pass::MatMulDecomposition::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    // Traverse nGraph Function in topological order
    bool is_graph_modfied = false;
    for (auto& node : f->get_ordered_ops()) {
        auto matmul = std::dynamic_pointer_cast<MatMul>(node);
        if (nullptr == matmul) {
            continue;
        }

        const Output<Node>& parent_a = matmul->input_value(0);
        const Output<Node>& parent_b = matmul->input_value(1);
        auto const_a = std::dynamic_pointer_cast<ngraph::op::Constant>(parent_a.get_node()->shared_from_this());
        auto const_b = std::dynamic_pointer_cast<ngraph::op::Constant>(parent_b.get_node()->shared_from_this());
        auto transpose_a = matmul->get_transpose_a();
        auto transpose_b = matmul->get_transpose_b();
        auto parent_a_shape = parent_a.get_shape();
        auto parent_b_shape = parent_b.get_shape();
        auto matmul_shape = matmul->get_output_shape(0);
        size_t N = (matmul_shape.size() > 3) ? matmul_shape[matmul_shape.size() - 4] : 1;
        size_t C = (matmul_shape.size() > 2) ? matmul_shape[matmul_shape.size() - 3] : 1;
        size_t H = (matmul_shape.size() > 1) ? matmul_shape[matmul_shape.size() - 2] : 1;
        size_t W = matmul_shape[matmul_shape.size() - 1];
        size_t N_a = (parent_a_shape.size() > 3) ? parent_a_shape[parent_a_shape.size() - 4] : 1;
        size_t C_a = (parent_a_shape.size() > 2) ? parent_a_shape[parent_a_shape.size() - 3] : 1;
        size_t H_a = (parent_a_shape.size() > 1) ? parent_a_shape[parent_a_shape.size() - 2] : 1;
        size_t W_a = parent_a_shape[parent_a_shape.size() - 1];
        size_t N_b = (parent_b_shape.size() > 3) ? parent_b_shape[parent_b_shape.size() - 4] : 1;
        size_t C_b = (parent_b_shape.size() > 2) ? parent_b_shape[parent_b_shape.size() - 3] : 1;
        size_t H_b = (parent_b_shape.size() > 1) ? parent_b_shape[parent_b_shape.size() - 2] : 1;
        size_t W_b = parent_b_shape[parent_b_shape.size() - 1];
        if (matmul_shape.size() == 1) {
            continue;
        } else if (N != 1) {
            continue;                                                              // Batch case not yet implemented
        } else if ((!transpose_a) && (const_b) && (parent_b_shape.size() == 2)) {  // factorization is straightforward
            OutputVector upstream;
            upstream.push_back(parent_a);
            if (parent_a_shape.size() > 2) {
                if (C_a == 1) {
                    auto new_reshape = std::make_shared<op::v1::Reshape>(
                        parent_a,
                        op::Constant::create(ngraph::element::i64, Shape{2}, {H_a, W_a})->output(0),
                        false);
                    upstream[0] = new_reshape->output(0);
                } else if (H_a == 1) {
                    auto new_reshape = std::make_shared<op::v1::Reshape>(
                        parent_a,
                        op::Constant::create(ngraph::element::i64, Shape{2}, {C_a, W_a})->output(0),
                        false);
                    upstream[0] = new_reshape->output(0);
                } else if (W_a == 1) {
                    auto new_reshape = std::make_shared<op::v1::Reshape>(
                        parent_a,
                        op::Constant::create(ngraph::element::i64, Shape{2}, {C_a, H_a})->output(0),
                        false);
                    upstream[0] = new_reshape->output(0);
                }
            }
            std::vector<size_t> split_lengths;
            OutputVector parts;
            if (transpose_b) {
                size_t H_b_left = H_b;
                while (H_b_left > 0) {
                    size_t H_b_part = (H_b_left > 8) ? 8 : H_b_left;
                    H_b_left -= H_b_part;
                    split_lengths.push_back(H_b_part);
                }
                size_t offset = 0;
                for (size_t k = 0; k < split_lengths.size(); k++) {
                    size_t H_new = W_b;
                    size_t W_new = split_lengths[k];
                    const float* weights_ptr = const_b->get_data_ptr<float>();
                    std::vector<float> new_weights(H_new * W_new, 0.0f);
                    float* new_weights_ptr = new_weights.data();
                    for (size_t i = 0; i < H_new; i++) {
                        for (size_t j = 0; j < W_new; j++) {
                            new_weights_ptr[i * W_new + j] = weights_ptr[(j + offset) * H_b + i];
                        }
                    }
                    auto new_weights_const =
                        op::Constant::create(ngraph::element::f32, Shape{W_b, split_lengths[k]}, new_weights);
                    auto new_matmul = std::make_shared<op::MatMul>(upstream[0], new_weights_const, false, false);
                    auto new_transpose =
                        std::make_shared<op::Transpose>(new_matmul->output(0),
                                                        op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                    parts.push_back(new_transpose->output(0));
                    offset += W_new;
                }
            } else {
                size_t W_b_left = W_b;
                while (W_b_left > 0) {
                    size_t W_b_part = (W_b_left > 8) ? 8 : W_b_left;
                    W_b_left -= W_b_part;
                    split_lengths.push_back(W_b_part);
                }
                size_t offset = 0;
                for (size_t k = 0; k < split_lengths.size(); k++) {
                    size_t H_new = H_b;
                    size_t W_new = split_lengths[k];
                    const float* weights_ptr = const_b->get_data_ptr<float>();
                    std::vector<float> new_weights(H_new * W_new, 0.0f);
                    float* new_weights_ptr = new_weights.data();
                    for (size_t i = 0; i < H_new; i++) {
                        for (size_t j = 0; j < W_new; j++) {
                            new_weights_ptr[i * W_new + j] = weights_ptr[i * W_b + j + offset];
                        }
                    }
                    auto new_weights_const =
                        op::Constant::create(ngraph::element::f32, Shape{H_b, split_lengths[k]}, new_weights);
                    auto new_matmul = std::make_shared<op::MatMul>(upstream[0], new_weights_const, false, false);
                    auto new_transpose =
                        std::make_shared<op::Transpose>(new_matmul->output(0),
                                                        op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                    parts.push_back(new_transpose->output(0));
                    offset += W_new;
                }
            }
            auto new_concat = std::make_shared<opset1::Concat>(parts, 0);
            // this is a big transpose requiring that the big transpose transformation be run after
            auto new_transpose =
                std::make_shared<op::Transpose>(new_concat->output(0),
                                                op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
            auto new_reshape = std::make_shared<op::v1::Reshape>(
                new_transpose->output(0),
                op::Constant::create(ngraph::element::i64, Shape{matmul_shape.size()}, matmul_shape)->output(0),
                false);
            ngraph::replace_node(matmul, new_reshape);
            is_graph_modfied = true;
        } else {
            if ((transpose_b && (H_b <= 8)) || (!transpose_b && (W_b <= 8))) {
                // If matmul is already compatible then skip it
                continue;
            }

            // if ((H > 8) && (W > 8)) {
            //    // If output is too large then a more complex work-around is required
            //    continue;
            //}

            // if ((!const_a) && ((!transpose_a && (H_a > 8)))) {
            //    // If A is not const and cannot be transposed by GNA then skip
            //    continue;
            //}

            if ((!const_b) && ((!transpose_b && (H_b > 8)))) {
                // If B is not const and cannot be transposed by GNA then
                // break it down into small blocks
                if (transpose_a || transpose_b) {  // not yet implemented
                    continue;
                }
                if (((H > 8) && (H % 8 != 0)) || (H_b % 8 != 0) || (W_b % 8 != 0)) {  // padding not yet implemented
                    continue;
                }

                OutputVector chunks_a, chunks_b;
                if (C > 1) {
                    auto reshape_a = std::make_shared<ngraph::opset1::Reshape>(
                        parent_a,
                        op::Constant::create(ngraph::element::i64, Shape{2}, {C_a, H_a * W_a})->output(0),
                        false);
                    auto reshape_b = std::make_shared<ngraph::opset1::Reshape>(
                        parent_b,
                        op::Constant::create(ngraph::element::i64, Shape{2}, {C_b, H_b * W_b})->output(0),
                        false);
                    auto split_a = std::make_shared<ngraph::opset1::Split>(
                        reshape_a,
                        ngraph::opset1::Constant::create(element::i64, Shape{}, {0}),
                        C_a);
                    auto split_b = std::make_shared<ngraph::opset1::Split>(
                        reshape_b,
                        ngraph::opset1::Constant::create(element::i64, Shape{}, {0}),
                        C_b);
                    for (size_t c = 0; c < C; c++) {
                        auto split_shape_a = parent_a.get_shape();
                        auto split_shape_b = parent_b.get_shape();
                        size_t dim1_a = split_shape_a[split_shape_a.size() - 2];
                        size_t dim2_a = split_shape_a[split_shape_a.size() - 1];
                        size_t dim1_b = split_shape_b[split_shape_b.size() - 2];
                        size_t dim2_b = split_shape_b[split_shape_b.size() - 1];
                        auto unreshape_a = std::make_shared<ngraph::opset1::Reshape>(
                            split_a->output(c),
                            op::Constant::create(ngraph::element::i64, Shape{2}, {dim1_a, dim2_a})->output(0),
                            false);
                        auto unreshape_b = std::make_shared<ngraph::opset1::Reshape>(
                            split_b->output(c),
                            op::Constant::create(ngraph::element::i64, Shape{2}, {dim1_b, dim2_b})->output(0),
                            false);
                        chunks_a.push_back(unreshape_a->output(0));
                        chunks_b.push_back(unreshape_b->output(0));
                    }
                } else {
                    auto reshape_a = std::make_shared<ngraph::opset1::Reshape>(
                        parent_a,
                        op::Constant::create(ngraph::element::i64, Shape{2}, {H_a, W_a})->output(0),
                        false);
                    auto reshape_b = std::make_shared<ngraph::opset1::Reshape>(
                        parent_b,
                        op::Constant::create(ngraph::element::i64, Shape{2}, {H_b, W_b})->output(0),
                        false);
                    chunks_a.push_back(reshape_a->output(0));
                    chunks_b.push_back(reshape_b->output(0));
                }

                bool abort = false;
                OutputVector product_parts;
                for (size_t c = 0; c < C; c++) {
                    std::shared_ptr<ov::Node> B_transpose = AdlBigTranspose2d(chunks_b[c]);
                    // if B can be transposed, more efficient method is possible
                    if (B_transpose != nullptr) {
                        std::vector<size_t> split_lengths;
                        size_t W_b_left = W_b;
                        while (W_b_left > 0) {
                            size_t W_b_part = (W_b_left > 8) ? 8 : W_b_left;
                            W_b_left -= W_b_part;
                            split_lengths.push_back(W_b_part);
                        }
                        const auto split_lengths_const = ngraph::opset1::Constant::create(element::i64,
                                                                                          Shape{split_lengths.size()},
                                                                                          split_lengths.data());
                        auto rowblock_b = std::make_shared<opset1::VariadicSplit>(
                            B_transpose->output(0),
                            ngraph::opset1::Constant::create(element::i64, Shape{}, {0}),
                            split_lengths_const);
                        OutputVector rowblock_c(split_lengths.size());
                        for (size_t i = 0; i < split_lengths.size(); i++) {
                            auto new_matmul =
                                std::make_shared<op::MatMul>(rowblock_b->output(i), chunks_a[c], false, true);
                            rowblock_c[i] = new_matmul->output(0);
                        }
                        auto new_concat = std::make_shared<ngraph::opset1::Concat>(rowblock_c, 0);
                        std::shared_ptr<ov::Node> result = AdlBigTranspose2d(new_concat->output(0));
                        if (nullptr == result) {
                            abort = true;
                        } else {
                            auto unsqueeze = std::make_shared<ngraph::opset1::Unsqueeze>(
                                result->output(0),
                                op::Constant::create(ngraph::element::i64, Shape{1}, {0})->output(0));
                            product_parts.push_back(unsqueeze->output(0));
                        }

                    } else {
                        // split B into row blocks of height 8
                        size_t H_b_div_8 = H_b / 8;
                        size_t W_b_div_8 = W_b / 8;
                        auto reshape = std::make_shared<ngraph::opset1::Reshape>(
                            chunks_b[c],
                            op::Constant::create(ngraph::element::i64, Shape{2}, {H_b, W_b})->output(0),
                            false);
                        auto rowblock_b = std::make_shared<ngraph::opset1::Split>(
                            reshape->output(0),
                            ngraph::opset1::Constant::create(element::i64, Shape{}, {0}),
                            H_b_div_8);
                        // transpose row blocks of B
                        std::vector<OutputVector> subblock_b;
                        for (size_t i = 0; i < H_b_div_8; i++) {
                            auto new_transpose = std::make_shared<op::Transpose>(
                                rowblock_b->output(i),
                                op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                            // split transposed row blocks B into 8x8 blocks
                            OutputVector block;
                            auto split_rowblock_b = std::make_shared<ngraph::opset1::Split>(
                                new_transpose->output(0),
                                ngraph::opset1::Constant::create(element::i64, Shape{}, {0}),
                                W_b_div_8);
                            for (size_t j = 0; j < W_b_div_8; j++) {
                                block.push_back(split_rowblock_b->output(j));
                            }
                            subblock_b.push_back(block);
                        }
                        // un-transpose all 8x8 blocks of B
                        for (size_t i = 0; i < H_b_div_8; i++) {
                            for (size_t j = 0; j < W_b_div_8; j++) {
                                auto new_transpose = std::make_shared<op::Transpose>(
                                    subblock_b[i][j],
                                    op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                                subblock_b[i][j] = new_transpose->output(0);
                            }
                        }
                        // concatenate into block columns of B
                        OutputVector colblock_b;
                        for (size_t j = 0; j < W_b_div_8; j++) {
                            OutputVector blocks;
                            for (size_t i = 0; i < H_b_div_8; i++) {
                                blocks.push_back(subblock_b[i][j]);
                            }
                            auto concat = std::make_shared<ngraph::opset1::Concat>(blocks, 0);
                            colblock_b.push_back(concat->output(0));
                        }
                        // multiply A by each block column of B
                        OutputVector colblock_c;
                        for (size_t j = 0; j < W_b_div_8; j++) {
                            auto new_matmul = std::make_shared<op::MatMul>(chunks_a[c], colblock_b[j], false, false);
                            colblock_c.push_back(new_matmul->output(0));
                        }
                        // split C into 8x8 blocks
                        size_t H_split = (H > 8) ? (H / 8) : 1;
                        std::vector<OutputVector> subblock_c(H_split);
                        if (H_split == 1) {
                            for (size_t j = 0; j < W_b_div_8; j++) {
                                subblock_c[0].push_back(colblock_c[j]);
                            }
                        } else {
                            auto block_shape = matmul->get_output_shape(0);
                            block_shape[block_shape.size() - 1] = 8;
                            block_shape[block_shape.size() - 2] = 8;
                            if (block_shape.size() > 2) {
                                block_shape[block_shape.size() - 3] = 1;
                            }
                            for (size_t j = 0; j < W_b_div_8; j++) {
                                auto split = std::make_shared<ngraph::opset1::Split>(
                                    colblock_c[j],
                                    ngraph::opset1::Constant::create(element::i64, Shape{}, {0}),
                                    H_split);
                                for (size_t i = 0; i < H_split; i++) {
                                    subblock_c[i].push_back(split->output(i));
                                }
                            }
                        }
                        // transpose each 8x8 block of C
                        for (size_t i = 0; i < H_split; i++) {
                            for (size_t j = 0; j < W_b_div_8; j++) {
                                auto new_transpose = std::make_shared<op::Transpose>(
                                    subblock_c[i][j],
                                    op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                                subblock_c[i][j] = new_transpose->output(0);
                            }
                        }
                        // concatenate into transposed block columns of C
                        OutputVector colblock_trans_c;
                        for (size_t i = 0; i < H_split; i++) {
                            OutputVector blocks;
                            for (size_t j = 0; j < W_b_div_8; j++) {
                                blocks.push_back(subblock_c[i][j]);
                            }
                            auto concat = std::make_shared<ngraph::opset1::Concat>(blocks, 0);
                            colblock_trans_c.push_back(concat->output(0));
                        }
                        // transpose block columns of C into block rows
                        OutputVector rowblock_c;
                        for (size_t i = 0; i < H_split; i++) {
                            auto new_transpose = std::make_shared<op::Transpose>(
                                colblock_trans_c[i],
                                op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
                            rowblock_c.push_back(new_transpose->output(0));
                        }
                        // contenate to form final product matrix C
                        if (rowblock_c.size() > 1) {
                            auto concat = std::make_shared<ngraph::opset1::Concat>(rowblock_c, 0);
                            auto unsqueeze = std::make_shared<ngraph::opset1::Unsqueeze>(
                                concat->output(0),
                                op::Constant::create(ngraph::element::i64, Shape{1}, {0})->output(0));
                            product_parts.push_back(unsqueeze->output(0));
                        } else {
                            auto unsqueeze = std::make_shared<ngraph::opset1::Unsqueeze>(
                                rowblock_c[0],
                                op::Constant::create(ngraph::element::i64, Shape{1}, {0})->output(0));
                            product_parts.push_back(unsqueeze->output(0));
                        }
                    }
                }

                if (abort) {
                    continue;  // transformation failed so abort
                }

                // contenate parts to form final product tensor
                if (C > 1) {
                    auto concat = std::make_shared<ngraph::opset1::Concat>(product_parts, 0);
                    auto unsqueeze = std::make_shared<ngraph::opset1::Unsqueeze>(
                        concat->output(0),
                        op::Constant::create(ngraph::element::i64, Shape{1}, {0})->output(0));
                    ngraph::replace_node(matmul, unsqueeze);
                } else {
                    auto unsqueeze = std::make_shared<ngraph::opset1::Unsqueeze>(
                        product_parts[0],
                        op::Constant::create(ngraph::element::i64, Shape{1}, {0})->output(0));
                    ngraph::replace_node(matmul, unsqueeze);
                }
                is_graph_modfied = true;
                continue;
            }

            if (C > 1) {
                auto axis = (matmul_shape.size() == 3) ? 0 : 1;
                const auto axis_node = ngraph::opset1::Constant::create(element::i64, Shape{}, {axis});
                const auto split_a = std::make_shared<ngraph::opset1::Split>(parent_a, axis_node, C);
                const auto split_b = std::make_shared<ngraph::opset1::Split>(parent_b, axis_node, C);
                auto new_transpose_a = !transpose_a;
                auto new_transpose_b = !transpose_b;
                OutputVector chunks;
                for (size_t c = 0; c < C; c++) {
                    auto new_matmul = std::make_shared<op::MatMul>(split_b->output(c),
                                                                   split_a->output(c),
                                                                   new_transpose_b,
                                                                   new_transpose_a);
                    std::shared_ptr<ngraph::Node> transpose_const;
                    if (matmul_shape.size() == 2) {
                        transpose_const = op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0});
                    } else if (matmul_shape.size() == 3) {
                        transpose_const = op::Constant::create(element::Type_t::i64, Shape{3}, {0, 2, 1});
                    } else {
                        transpose_const = op::Constant::create(element::Type_t::i64, Shape{4}, {0, 1, 3, 2});
                    }
                    auto new_transpose = std::make_shared<op::Transpose>(new_matmul->output(0), transpose_const);
                    chunks.push_back(new_transpose->output(0));
                }
                auto new_concat = std::make_shared<ngraph::opset1::Concat>(chunks, axis);
                ngraph::replace_node(matmul, new_concat);

            } else {
                auto new_transpose_a = !transpose_a;
                auto new_transpose_b = !transpose_b;
                auto new_matmul = std::make_shared<op::MatMul>(parent_b, parent_a, new_transpose_b, new_transpose_a);
                std::shared_ptr<ngraph::Node> transpose_const;
                if (matmul_shape.size() == 2) {
                    transpose_const = op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0});
                } else if (matmul_shape.size() == 3) {
                    transpose_const = op::Constant::create(element::Type_t::i64, Shape{3}, {0, 2, 1});
                } else {
                    transpose_const = op::Constant::create(element::Type_t::i64, Shape{4}, {0, 1, 3, 2});
                }
                auto new_transpose = std::make_shared<op::Transpose>(new_matmul->output(0), transpose_const);
                ngraph::replace_node(matmul, new_transpose);
            }
            is_graph_modfied = true;
        }
    }
    return is_graph_modfied;
}
