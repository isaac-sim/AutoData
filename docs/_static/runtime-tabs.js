// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Sphinx-design synchronizes label clicks; keyboard radio changes need the same event.
document.addEventListener("change", (event) => {
  const input = event.target;
  if (!(input instanceof HTMLInputElement) || input.type !== "radio" || !input.checked) return;
  const label = input.nextElementSibling;
  if (label?.matches('.sd-tab-label[data-sync-group="autodata-runtime"]')) label.click();
});
