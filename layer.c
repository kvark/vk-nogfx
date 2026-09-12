/* Implicit Vulkan layer: present a compute-only queue family as index 0 and
 * remap CreateDevice/GetDeviceQueue/CreateCommandPool so apps that hardcode
 * family 0 do not hit a dummy GRAPHICS ring. Enablement is the caller's job. */
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define LAYER_NAME "VK_LAYER_NOGFX_compute_first"
#define MAX_FAMILIES 16
#define MAX_OBJECTS 32

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *pName);
VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *pName);
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *pVersionStruct);

struct family_map {
    uint32_t count;
    uint32_t to_real[MAX_FAMILIES];
    int active;
};

struct instance_data {
    VkInstance instance;
    PFN_vkGetInstanceProcAddr gipa;
    PFN_vkDestroyInstance DestroyInstance;
    PFN_vkCreateDevice CreateDevice;
    PFN_vkGetDeviceProcAddr GetDeviceProcAddr;
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties2 GetPhysicalDeviceQueueFamilyProperties2;
    PFN_vkGetPhysicalDeviceSurfaceSupportKHR GetPhysicalDeviceSurfaceSupportKHR;
};

struct device_data {
    VkDevice device;
    struct family_map map;
    PFN_vkGetDeviceProcAddr gdpa;
    PFN_vkGetDeviceQueue GetDeviceQueue;
    PFN_vkGetDeviceQueue2 GetDeviceQueue2;
    PFN_vkCreateCommandPool CreateCommandPool;
    PFN_vkDestroyDevice DestroyDevice;
};

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static struct instance_data instances[MAX_OBJECTS];
static struct device_data devices[MAX_OBJECTS];
static struct {
    VkPhysicalDevice handle;
    struct family_map map;
} phys[MAX_OBJECTS];

static int debug_enabled(void) {
    const char *e = getenv("VK_LAYER_NOGFX_DEBUG");
    return e && e[0] && e[0] != '0';
}

static struct instance_data *instance_slot(VkInstance inst, int create) {
    for (int i = 0; i < MAX_OBJECTS; i++) {
        if (instances[i].instance == inst)
            return &instances[i];
    }
    if (!create)
        return NULL;
    for (int i = 0; i < MAX_OBJECTS; i++) {
        if (!instances[i].instance) {
            memset(&instances[i], 0, sizeof(instances[i]));
            instances[i].instance = inst;
            return &instances[i];
        }
    }
    return NULL;
}

static struct device_data *device_slot(VkDevice dev, int create) {
    for (int i = 0; i < MAX_OBJECTS; i++) {
        if (devices[i].device == dev)
            return &devices[i];
    }
    if (!create)
        return NULL;
    for (int i = 0; i < MAX_OBJECTS; i++) {
        if (!devices[i].device) {
            memset(&devices[i], 0, sizeof(devices[i]));
            devices[i].device = dev;
            return &devices[i];
        }
    }
    return NULL;
}

static struct family_map *phys_map(VkPhysicalDevice phd, int create) {
    for (int i = 0; i < MAX_OBJECTS; i++) {
        if (phys[i].handle == phd)
            return &phys[i].map;
    }
    if (!create)
        return NULL;
    for (int i = 0; i < MAX_OBJECTS; i++) {
        if (!phys[i].handle) {
            phys[i].handle = phd;
            memset(&phys[i].map, 0, sizeof(phys[i].map));
            return &phys[i].map;
        }
    }
    return NULL;
}

static uint32_t to_real(const struct family_map *map, uint32_t app) {
    if (!map || !map->active || app >= MAX_FAMILIES)
        return app;
    return map->to_real[app];
}

static void build_map(struct instance_data *inst, VkPhysicalDevice phd, struct family_map *map) {
    VkQueueFamilyProperties props[MAX_FAMILIES];
    uint32_t count = MAX_FAMILIES;
    VkPhysicalDeviceProperties devp;
    inst->GetPhysicalDeviceProperties(phd, &devp);
    inst->GetPhysicalDeviceQueueFamilyProperties(phd, &count, props);
    if (count > MAX_FAMILIES)
        count = MAX_FAMILIES;
    map->count = count;
    map->active = 0;
    for (uint32_t i = 0; i < MAX_FAMILIES; i++)
        map->to_real[i] = i;
    if (count == 0)
        return;
    uint32_t compute_only = UINT32_MAX;
    for (uint32_t i = 0; i < count; i++) {
        VkQueueFlags f = props[i].queueFlags;
        if ((f & VK_QUEUE_COMPUTE_BIT) && !(f & VK_QUEUE_GRAPHICS_BIT)) {
            compute_only = i;
            break;
        }
    }
    if (compute_only == UINT32_MAX || compute_only == 0)
        return;
    if (!(props[0].queueFlags & VK_QUEUE_GRAPHICS_BIT))
        return;
    map->to_real[0] = compute_only;
    map->to_real[compute_only] = 0;
    map->active = 1;
    if (debug_enabled()) {
        fprintf(stderr, "[%s] %s: app family 0 -> driver family %u (compute-only)\n", LAYER_NAME,
                devp.deviceName, compute_only);
    }
}

static void ensure_map(struct instance_data *inst, VkPhysicalDevice phd) {
    struct family_map *map = phys_map(phd, 1);
    if (!map || map->count)
        return;
    build_map(inst, phd, map);
}

static void apply_app_order(const struct family_map *map, uint32_t count,
                            VkQueueFamilyProperties *props) {
    VkQueueFamilyProperties real[MAX_FAMILIES];
    if (!map || !map->active || !props || count == 0)
        return;
    if (count > MAX_FAMILIES)
        count = MAX_FAMILIES;
    memcpy(real, props, count * sizeof(*props));
    for (uint32_t i = 0; i < count; i++)
        props[i] = real[map->to_real[i]];
}

static VkLayerInstanceCreateInfo *find_instance_link(const VkInstanceCreateInfo *info) {
    const VkLayerInstanceCreateInfo *chain = (const void *)info->pNext;
    while (chain) {
        if (chain->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
            chain->function == VK_LAYER_LINK_INFO)
            return (VkLayerInstanceCreateInfo *)chain;
        chain = (const void *)chain->pNext;
    }
    return NULL;
}

static VkLayerDeviceCreateInfo *find_device_link(const VkDeviceCreateInfo *info) {
    const VkLayerDeviceCreateInfo *chain = (const void *)info->pNext;
    while (chain) {
        if (chain->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO &&
            chain->function == VK_LAYER_LINK_INFO)
            return (VkLayerDeviceCreateInfo *)chain;
        chain = (const void *)chain->pNext;
    }
    return NULL;
}

static struct instance_data *instance_from_phys(VkPhysicalDevice phd) {
    (void)phd;
    for (int i = 0; i < MAX_OBJECTS; i++) {
        if (instances[i].instance)
            return &instances[i];
    }
    return NULL;
}

static VKAPI_ATTR VkResult VKAPI_CALL
layer_CreateInstance(const VkInstanceCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator,
                     VkInstance *pInstance) {
    VkLayerInstanceCreateInfo *link = find_instance_link(pCreateInfo);
    if (!link)
        return VK_ERROR_INITIALIZATION_FAILED;
    PFN_vkGetInstanceProcAddr next_gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;
    PFN_vkCreateInstance create = (PFN_vkCreateInstance)next_gipa(VK_NULL_HANDLE, "vkCreateInstance");
    VkResult r = create(pCreateInfo, pAllocator, pInstance);
    if (r != VK_SUCCESS)
        return r;
    pthread_mutex_lock(&lock);
    struct instance_data *inst = instance_slot(*pInstance, 1);
    if (inst) {
        inst->gipa = next_gipa;
        inst->DestroyInstance = (PFN_vkDestroyInstance)next_gipa(*pInstance, "vkDestroyInstance");
        inst->CreateDevice = (PFN_vkCreateDevice)next_gipa(*pInstance, "vkCreateDevice");
        inst->GetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)next_gipa(*pInstance, "vkGetDeviceProcAddr");
        inst->GetPhysicalDeviceProperties =
            (PFN_vkGetPhysicalDeviceProperties)next_gipa(*pInstance, "vkGetPhysicalDeviceProperties");
        inst->GetPhysicalDeviceQueueFamilyProperties =
            (PFN_vkGetPhysicalDeviceQueueFamilyProperties)next_gipa(
                *pInstance, "vkGetPhysicalDeviceQueueFamilyProperties");
        inst->GetPhysicalDeviceQueueFamilyProperties2 =
            (PFN_vkGetPhysicalDeviceQueueFamilyProperties2)next_gipa(
                *pInstance, "vkGetPhysicalDeviceQueueFamilyProperties2");
        if (!inst->GetPhysicalDeviceQueueFamilyProperties2)
            inst->GetPhysicalDeviceQueueFamilyProperties2 =
                (PFN_vkGetPhysicalDeviceQueueFamilyProperties2)next_gipa(
                    *pInstance, "vkGetPhysicalDeviceQueueFamilyProperties2KHR");
        inst->GetPhysicalDeviceSurfaceSupportKHR =
            (PFN_vkGetPhysicalDeviceSurfaceSupportKHR)next_gipa(
                *pInstance, "vkGetPhysicalDeviceSurfaceSupportKHR");
    }
    pthread_mutex_unlock(&lock);
    return r;
}

static VKAPI_ATTR void VKAPI_CALL
layer_DestroyInstance(VkInstance instance, const VkAllocationCallbacks *pAllocator) {
    pthread_mutex_lock(&lock);
    struct instance_data *inst = instance_slot(instance, 0);
    PFN_vkDestroyInstance destroy = inst ? inst->DestroyInstance : NULL;
    if (inst)
        memset(inst, 0, sizeof(*inst));
    pthread_mutex_unlock(&lock);
    if (destroy)
        destroy(instance, pAllocator);
}

static VKAPI_ATTR void VKAPI_CALL
layer_GetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice physicalDevice, uint32_t *pCount,
                                             VkQueueFamilyProperties *pProperties) {
    pthread_mutex_lock(&lock);
    struct instance_data *inst = instance_from_phys(physicalDevice);
    if (!inst || !inst->GetPhysicalDeviceQueueFamilyProperties) {
        pthread_mutex_unlock(&lock);
        return;
    }
    inst->GetPhysicalDeviceQueueFamilyProperties(physicalDevice, pCount, pProperties);
    if (pProperties && pCount) {
        ensure_map(inst, physicalDevice);
        apply_app_order(phys_map(physicalDevice, 0), *pCount, pProperties);
    }
    pthread_mutex_unlock(&lock);
}

static VKAPI_ATTR void VKAPI_CALL
layer_GetPhysicalDeviceQueueFamilyProperties2(VkPhysicalDevice physicalDevice, uint32_t *pCount,
                                              VkQueueFamilyProperties2 *pProperties) {
    pthread_mutex_lock(&lock);
    struct instance_data *inst = instance_from_phys(physicalDevice);
    if (!inst) {
        pthread_mutex_unlock(&lock);
        return;
    }
    if (inst->GetPhysicalDeviceQueueFamilyProperties2)
        inst->GetPhysicalDeviceQueueFamilyProperties2(physicalDevice, pCount, pProperties);
    else if (pProperties && pCount) {
        VkQueueFamilyProperties tmp[MAX_FAMILIES];
        uint32_t n = *pCount;
        inst->GetPhysicalDeviceQueueFamilyProperties(physicalDevice, &n, tmp);
        if (n > *pCount)
            n = *pCount;
        for (uint32_t i = 0; i < n; i++)
            pProperties[i].queueFamilyProperties = tmp[i];
        *pCount = n;
    } else {
        inst->GetPhysicalDeviceQueueFamilyProperties(physicalDevice, pCount, NULL);
    }
    if (pProperties && pCount) {
        ensure_map(inst, physicalDevice);
        struct family_map *map = phys_map(physicalDevice, 0);
        if (map && map->active) {
            VkQueueFamilyProperties tmp[MAX_FAMILIES];
            uint32_t n = *pCount > MAX_FAMILIES ? MAX_FAMILIES : *pCount;
            for (uint32_t i = 0; i < n; i++)
                tmp[i] = pProperties[i].queueFamilyProperties;
            apply_app_order(map, n, tmp);
            for (uint32_t i = 0; i < n; i++)
                pProperties[i].queueFamilyProperties = tmp[i];
        }
    }
    pthread_mutex_unlock(&lock);
}

static VKAPI_ATTR VkResult VKAPI_CALL
layer_GetPhysicalDeviceSurfaceSupportKHR(VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex,
                                         VkSurfaceKHR surface, VkBool32 *pSupported) {
    pthread_mutex_lock(&lock);
    struct instance_data *inst = instance_from_phys(physicalDevice);
    ensure_map(inst, physicalDevice);
    struct family_map *map = phys_map(physicalDevice, 0);
    uint32_t real = to_real(map, queueFamilyIndex);
    PFN_vkGetPhysicalDeviceSurfaceSupportKHR fn =
        inst ? inst->GetPhysicalDeviceSurfaceSupportKHR : NULL;
    pthread_mutex_unlock(&lock);
    if (!fn)
        return VK_ERROR_INITIALIZATION_FAILED;
    return fn(physicalDevice, real, surface, pSupported);
}

static VKAPI_ATTR VkResult VKAPI_CALL
layer_CreateDevice(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo *pCreateInfo,
                   const VkAllocationCallbacks *pAllocator, VkDevice *pDevice) {
    pthread_mutex_lock(&lock);
    struct instance_data *inst = instance_from_phys(physicalDevice);
    if (!inst) {
        pthread_mutex_unlock(&lock);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    ensure_map(inst, physicalDevice);
    struct family_map map = {0};
    struct family_map *stored = phys_map(physicalDevice, 0);
    if (stored)
        map = *stored;
    VkLayerDeviceCreateInfo *link = find_device_link(pCreateInfo);
    PFN_vkGetDeviceProcAddr next_gdpa = NULL;
    if (link) {
        next_gdpa = link->u.pLayerInfo->pfnNextGetDeviceProcAddr;
        link->u.pLayerInfo = link->u.pLayerInfo->pNext;
    }
    VkDeviceQueueCreateInfo queues[MAX_FAMILIES];
    VkDeviceCreateInfo ci = *pCreateInfo;
    if (map.active && pCreateInfo->queueCreateInfoCount) {
        uint32_t n = pCreateInfo->queueCreateInfoCount;
        if (n > MAX_FAMILIES)
            n = MAX_FAMILIES;
        memcpy(queues, pCreateInfo->pQueueCreateInfos, n * sizeof(queues[0]));
        for (uint32_t i = 0; i < n; i++)
            queues[i].queueFamilyIndex = to_real(&map, queues[i].queueFamilyIndex);
        ci.pQueueCreateInfos = queues;
        ci.queueCreateInfoCount = n;
    }
    PFN_vkCreateDevice create = inst->CreateDevice;
    pthread_mutex_unlock(&lock);

    VkResult r = create(physicalDevice, &ci, pAllocator, pDevice);
    if (r != VK_SUCCESS)
        return r;

    pthread_mutex_lock(&lock);
    struct device_data *dev = device_slot(*pDevice, 1);
    if (dev) {
        dev->map = map;
        if (!next_gdpa && inst->GetDeviceProcAddr)
            next_gdpa = inst->GetDeviceProcAddr;
        dev->gdpa = next_gdpa;
        if (next_gdpa) {
            dev->GetDeviceQueue = (PFN_vkGetDeviceQueue)next_gdpa(*pDevice, "vkGetDeviceQueue");
            dev->GetDeviceQueue2 = (PFN_vkGetDeviceQueue2)next_gdpa(*pDevice, "vkGetDeviceQueue2");
            dev->CreateCommandPool =
                (PFN_vkCreateCommandPool)next_gdpa(*pDevice, "vkCreateCommandPool");
            dev->DestroyDevice = (PFN_vkDestroyDevice)next_gdpa(*pDevice, "vkDestroyDevice");
        }
    }
    pthread_mutex_unlock(&lock);
    return r;
}

static VKAPI_ATTR void VKAPI_CALL
layer_GetDeviceQueue(VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex, VkQueue *pQueue) {
    pthread_mutex_lock(&lock);
    struct device_data *dev = device_slot(device, 0);
    uint32_t real = to_real(dev ? &dev->map : NULL, queueFamilyIndex);
    PFN_vkGetDeviceQueue fn = dev ? dev->GetDeviceQueue : NULL;
    pthread_mutex_unlock(&lock);
    if (fn)
        fn(device, real, queueIndex, pQueue);
}

static VKAPI_ATTR void VKAPI_CALL
layer_GetDeviceQueue2(VkDevice device, const VkDeviceQueueInfo2 *pQueueInfo, VkQueue *pQueue) {
    pthread_mutex_lock(&lock);
    struct device_data *dev = device_slot(device, 0);
    VkDeviceQueueInfo2 info = *pQueueInfo;
    info.queueFamilyIndex = to_real(dev ? &dev->map : NULL, pQueueInfo->queueFamilyIndex);
    PFN_vkGetDeviceQueue2 fn = dev ? dev->GetDeviceQueue2 : NULL;
    pthread_mutex_unlock(&lock);
    if (fn)
        fn(device, &info, pQueue);
}

static VKAPI_ATTR VkResult VKAPI_CALL
layer_CreateCommandPool(VkDevice device, const VkCommandPoolCreateInfo *pCreateInfo,
                        const VkAllocationCallbacks *pAllocator, VkCommandPool *pCommandPool) {
    pthread_mutex_lock(&lock);
    struct device_data *dev = device_slot(device, 0);
    VkCommandPoolCreateInfo info = *pCreateInfo;
    info.queueFamilyIndex = to_real(dev ? &dev->map : NULL, pCreateInfo->queueFamilyIndex);
    PFN_vkCreateCommandPool fn = dev ? dev->CreateCommandPool : NULL;
    pthread_mutex_unlock(&lock);
    if (!fn)
        return VK_ERROR_INITIALIZATION_FAILED;
    return fn(device, &info, pAllocator, pCommandPool);
}

static VKAPI_ATTR void VKAPI_CALL
layer_DestroyDevice(VkDevice device, const VkAllocationCallbacks *pAllocator) {
    pthread_mutex_lock(&lock);
    struct device_data *dev = device_slot(device, 0);
    PFN_vkDestroyDevice destroy = dev ? dev->DestroyDevice : NULL;
    if (dev)
        memset(dev, 0, sizeof(*dev));
    pthread_mutex_unlock(&lock);
    if (destroy)
        destroy(device, pAllocator);
}

static PFN_vkVoidFunction find_instance_proc(const char *name) {
    if (!strcmp(name, "vkCreateInstance"))
        return (PFN_vkVoidFunction)layer_CreateInstance;
    if (!strcmp(name, "vkDestroyInstance"))
        return (PFN_vkVoidFunction)layer_DestroyInstance;
    if (!strcmp(name, "vkCreateDevice"))
        return (PFN_vkVoidFunction)layer_CreateDevice;
    if (!strcmp(name, "vkGetDeviceProcAddr"))
        return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
    if (!strcmp(name, "vkGetInstanceProcAddr"))
        return (PFN_vkVoidFunction)vkGetInstanceProcAddr;
    if (!strcmp(name, "vkGetPhysicalDeviceQueueFamilyProperties"))
        return (PFN_vkVoidFunction)layer_GetPhysicalDeviceQueueFamilyProperties;
    if (!strcmp(name, "vkGetPhysicalDeviceQueueFamilyProperties2") ||
        !strcmp(name, "vkGetPhysicalDeviceQueueFamilyProperties2KHR"))
        return (PFN_vkVoidFunction)layer_GetPhysicalDeviceQueueFamilyProperties2;
    if (!strcmp(name, "vkGetPhysicalDeviceSurfaceSupportKHR"))
        return (PFN_vkVoidFunction)layer_GetPhysicalDeviceSurfaceSupportKHR;
    if (!strcmp(name, "vkNegotiateLoaderLayerInterfaceVersion"))
        return (PFN_vkVoidFunction)vkNegotiateLoaderLayerInterfaceVersion;
    return NULL;
}

static PFN_vkVoidFunction find_device_proc(const char *name) {
    if (!strcmp(name, "vkGetDeviceProcAddr"))
        return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
    if (!strcmp(name, "vkGetDeviceQueue"))
        return (PFN_vkVoidFunction)layer_GetDeviceQueue;
    if (!strcmp(name, "vkGetDeviceQueue2"))
        return (PFN_vkVoidFunction)layer_GetDeviceQueue2;
    if (!strcmp(name, "vkCreateCommandPool"))
        return (PFN_vkVoidFunction)layer_CreateCommandPool;
    if (!strcmp(name, "vkDestroyDevice"))
        return (PFN_vkVoidFunction)layer_DestroyDevice;
    return NULL;
}

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetInstanceProcAddr(VkInstance instance, const char *pName) {
    PFN_vkVoidFunction layer_fn = find_instance_proc(pName);
    if (layer_fn)
        return layer_fn;
    layer_fn = find_device_proc(pName);
    if (layer_fn)
        return layer_fn;
    pthread_mutex_lock(&lock);
    struct instance_data *inst = instance ? instance_slot(instance, 0) : NULL;
    PFN_vkGetInstanceProcAddr gipa = inst ? inst->gipa : NULL;
    pthread_mutex_unlock(&lock);
    if (!gipa)
        return NULL;
    return gipa(instance, pName);
}

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vkGetDeviceProcAddr(VkDevice device, const char *pName) {
    PFN_vkVoidFunction layer_fn = find_device_proc(pName);
    if (layer_fn)
        return layer_fn;
    pthread_mutex_lock(&lock);
    struct device_data *dev = device ? device_slot(device, 0) : NULL;
    PFN_vkGetDeviceProcAddr gdpa = dev ? dev->gdpa : NULL;
    pthread_mutex_unlock(&lock);
    if (!gdpa)
        return NULL;
    return gdpa(device, pName);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *pVersionStruct) {
    if (!pVersionStruct || pVersionStruct->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (pVersionStruct->loaderLayerInterfaceVersion < MIN_SUPPORTED_LOADER_LAYER_INTERFACE_VERSION)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (pVersionStruct->loaderLayerInterfaceVersion > CURRENT_LOADER_LAYER_INTERFACE_VERSION)
        pVersionStruct->loaderLayerInterfaceVersion = CURRENT_LOADER_LAYER_INTERFACE_VERSION;
    pVersionStruct->pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
    pVersionStruct->pfnGetDeviceProcAddr = vkGetDeviceProcAddr;
    pVersionStruct->pfnGetPhysicalDeviceProcAddr = NULL;
    return VK_SUCCESS;
}
