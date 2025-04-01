# SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

OUTNAME_RELEASE = sample_non_zero_plugin
OUTNAME_DEBUG   = sample_non_zero_plugin_debug
EXTRA_DIRECTORIES = ../common ../utils
SAMPLE_DIR_NAME = $(shell basename $(dir $(abspath $(firstword $(MAKEFILE_LIST)))))
MAKEFILE ?= ../Makefile.config

ifneq ($(CUDA_INSTALL_DIR), )
	CUDA_INST_DIR ?= $(CUDA_INSTALL_DIR)
endif
CUDA_INST_DIR ?= /usr/local/cuda
CUDA_VER := $(shell $(CUDA_INST_DIR)/bin/nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')

define cuda-ge
    CUDA_VER_GE_$(1)_$(2) := 0
    ifeq ($$(lastword $$(sort $(3) $(1).$(2))), $(3))
        CUDA_VER_GE_$(1)_$(2) := 1
    endif

    CUDA_GE_$(1)_$(2) ?= $$(CUDA_VER_GE_$(1)_$(2))

endef

$(eval $(foreach minor,$(shell seq 0 9),$(call cuda-ge,11,$(minor),$(CUDA_VER))))

SAMPLE_SMS = 75
ifeq ($(CUDA_GE_11_0), 1)
    SAMPLE_SMS += 80
endif

ifeq ($(CUDA_GE_11_1), 1)
    SAMPLE_SMS += 86
endif

ifeq ($(CUDA_GE_11_4), 1)
    SAMPLE_SMS += 87
endif

ifeq ($(CUDA_GE_11_8), 1)
    SAMPLE_SMS += 89 90
endif

# Non-hardware forward compatible SMs
NON_HFC_SMS := 89 90

ifeq ($(GENCODES),)
    $(foreach sm,$(SAMPLE_SMS),$(eval GENCODES += -gencode arch=compute_$(sm),code=sm_$(sm)))
    HFC_SMS := $(filter-out $(NON_HFC_SMS),$(SAMPLE_SMS))
    # Generate PTX with the highest supported FC SM
    GEN_PTX_SM := $(lastword $(sort $(HFC_SMS)))
    GENCODES += -gencode arch=compute_$(GEN_PTX_SM),code=compute_$(GEN_PTX_SM)
endif

include $(MAKEFILE)
