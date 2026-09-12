# VK_LAYER_NOGFX_compute_first

System-wide **implicit** Vulkan layer. If family 0 advertises GRAPHICS and a
later family is compute-only, the layer presents that compute family as index 0
and remaps `vkCreateDevice` / `vkGetDeviceQueue` / `vkCreateCommandPool`.

It does not try to recognize a GPU. Enable it only on machines where family 0
is not the queue you want.

```bash
make
sudo make install
```

The implicit JSON is installed to `/usr/share/vulkan/implicit_layer.d/` with an
absolute `library_path`. Disable with:

```bash
export DISABLE_VK_LAYER_NOGFX_compute_first=1
```

Debug remaps with `VK_LAYER_NOGFX_DEBUG=1`.
