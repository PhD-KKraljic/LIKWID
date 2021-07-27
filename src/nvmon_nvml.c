/*
 * =======================================================================================
 *
 *      Filename:  nvmon.c
 *
 *      Description:  Main implementation of the performance monitoring module
 *                    for NVIDIA GPUs
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tg), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2016 RRZE, University Erlangen-Nuremberg
 *
 *      This program is free software: you can redistribute it and/or modify it under
 *      the terms of the GNU General Public License as published by the Free Software
 *      Foundation, either version 3 of the License, or (at your option) any later
 *      version.
 *
 *      This program is distributed in the hope that it will be useful, but WITHOUT ANY
 *      WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 *      PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 *      You should have received a copy of the GNU General Public License along with
 *      this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * =======================================================================================
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <unistd.h>
#include <sys/types.h>

#include <dlfcn.h>
#include <nvml.h>
#include <cupti.h>

#include <likwid.h>
#include <error.h>
#include <nvmon_types.h>
#include <libnvctr_types.h>

typedef enum {
    FEATURE_CLOCK_INFO          = 1,
    FEATURE_ECC_LOCAL_ERRORS    = 2,
    FEATURE_FAN_SPEED           = 4,
    FEATURE_MAX_CLOCK           = 8,
    FEATURE_MEMORY_INFO         = 16,
    FEATURE_PERF_STATES         = 32,
    FEATURE_POWER               = 64,
    FEATURE_TEMP                = 128,
    FEATURE_ECC_TOTAL_ERRORS    = 256,
    FEATURE_UTILIZATION         = 512,
    FEATURE_POWER_MANAGEMENT    = 1024,
    FEATURE_NVML_POWER_MANAGEMENT_LIMIT_CONSTRAINT_MIN = 2048,
    FEATURE_NVML_POWER_MANAGEMENT_LIMIT_CONSTRAINT_MAX = 4096,
} NvmlFeature;

typedef struct {
    double fullValue;
    double lastValue;
} NvmlEventResult;

struct NvmlEvent_struct;
typedef int (*NvmlMeasureFunc)(nvmlDevice_t device, struct NvmlEvent_struct* event, NvmlEventResult* result);

#define LIKWID_NVML_NAME_LEN 40
typedef struct NvmlEvent_struct {
    char name[LIKWID_NVML_NAME_LEN];
    NvmlMeasureFunc measureFunc;
    unsigned int variant;
} NvmlEvent;

typedef struct {
    int numEvents;
    NvmlEvent* events;
    NvmlEventResult* results;
} NvmlEventSet;

typedef struct {
    NvmonDevice* nvDevice;
    nvmlDevice_t nvmlDevice;

    int numAllEvents;
    NvmlEvent* allEvents;

    int activeEventSet;
    int numEventSets;
    NvmlEventSet* eventSets;

    uint32_t features;

    // Timestamps in ns
    struct {
        uint64_t start;
        uint64_t read;
        uint64_t stop;
    } time;
} NvmlDevice;

typedef struct {
    int numDevices;
    NvmlDevice* devices;
} NvmlContext;


// Variables
static int nvml_initialized = 0;
static void* dl_nvml = NULL;
static void* dl_cupti = NULL;
static NvmlContext nvmlContext;


// Macros
#define FREE_IF_NOT_NULL(x) if (x) { free(x); }
#define DLSYM_AND_CHECK( dllib, name ) name##_ptr = dlsym( dllib, #name ); if ( dlerror() != NULL ) { return -1; }
#define NVML_CALL(call, args, handleerror)                                            \
    do {                                                                           \
        nvmlReturn_t _status = (*call##_ptr)args;                                         \
        if (_status != NVML_SUCCESS) {                                            \
            fprintf(stderr, "Error: function %s failed with error %d.\n", #call, _status);                    \
            handleerror;                                                             \
        }                                                                          \
    } while (0)
#define CUPTI_CALL(call, args, handleerror)                                            \
    do {                                                                \
        CUptiResult _status = (*call##_ptr)args;                                  \
        if (_status != CUPTI_SUCCESS) {                                 \
            const char *errstr;                                         \
            (*cuptiGetResultString)(_status, &errstr);               \
            fprintf(stderr, "Error: function %s failed with error %s.\n", #call, errstr); \
            handleerror;                                                \
        }                                                               \
    } while (0)

// NVML function declarations
#define NVMLWEAK __attribute__(( weak ))
#define DECLAREFUNC_NVML(funcname, funcsig) nvmlReturn_t NVMLWEAK funcname funcsig;  nvmlReturn_t ( *funcname##_ptr ) funcsig;

DECLAREFUNC_NVML(nvmlInit_v2, (void));
DECLAREFUNC_NVML(nvmlShutdown, (void));
DECLAREFUNC_NVML(nvmlDeviceGetHandleByIndex_v2, (unsigned int  index, nvmlDevice_t* device));
DECLAREFUNC_NVML(nvmlDeviceGetClockInfo, (nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock));

// CUPTI function declarations
#define CUPTIWEAK __attribute__(( weak ))
#define DECLAREFUNC_CUPTI(funcname, funcsig) CUptiResult CUPTIWEAK funcname funcsig;  CUptiResult( *funcname##_ptr ) funcsig;

DECLAREFUNC_CUPTI(cuptiGetTimestamp, (uint64_t * timestamp));
DECLAREFUNC_CUPTI(cuptiGetResultString, (CUptiResult result, const char **str));


// ----------------------------------------------------
//   Wrapper functions
// ----------------------------------------------------

static int
_nvml_wrapper_getClockInfo(nvmlDevice_t device, NvmlEvent* event, NvmlEventResult* result)
{
    unsigned int clock;

    NVML_CALL(nvmlDeviceGetClockInfo, (device, event->variant, &clock), return -1);
    result->fullValue += clock;
    result->lastValue = clock;

    return 0;
}


// ----------------------------------------------------
//   Helper functions
// ----------------------------------------------------

static int
_nvml_linkLibraries()
{
    // Load NVML libary and link functions
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Init NVML Libaries);
    dl_nvml = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dl_nvml || dlerror() != NULL)
    {
        fprintf(stderr, "NVML library libnvidia-ml.so not found.");
        return -1;
    }

    DLSYM_AND_CHECK(dl_nvml, nvmlInit_v2);
    DLSYM_AND_CHECK(dl_nvml, nvmlShutdown);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetHandleByIndex_v2);
    DLSYM_AND_CHECK(dl_nvml, nvmlDeviceGetClockInfo);

    // Load CUPTI library and link functions
    GPUDEBUG_PRINT(DEBUGLEV_DEVELOP, Init NVML Libaries);
    dl_cupti = dlopen("libcupti.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dl_cupti || dlerror() != NULL)
    {
        fprintf(stderr, "CUPTI library libcupti.so not found.");
        return -1;
    }

    DLSYM_AND_CHECK(dl_cupti, cuptiGetTimestamp);
    DLSYM_AND_CHECK(dl_cupti, cuptiGetResultString);

    return 0;
}


static int
_nvml_getEventsForDevice(NvmlDevice* device)
{
    NvmlEvent* event = device->allEvents;

    if (device->features & FEATURE_CLOCK_INFO)
    {
        snprintf(event->name, LIKWID_NVML_NAME_LEN, "CLOCK_GRAPHICS");
        event->measureFunc = &_nvml_wrapper_getClockInfo;
        event->variant = NVML_CLOCK_GRAPHICS;
        event++;

        snprintf(event->name, LIKWID_NVML_NAME_LEN, "CLOCK_SM");
        event->measureFunc = &_nvml_wrapper_getClockInfo;
        event->variant = NVML_CLOCK_SM;
        event++;

        snprintf(event->name, LIKWID_NVML_NAME_LEN, "CLOCK_MEM");
        event->measureFunc = &_nvml_wrapper_getClockInfo;
        event->variant = NVML_CLOCK_MEM;
        event++;

        snprintf(event->name, LIKWID_NVML_NAME_LEN, "CLOCK_VIDEO");
        event->measureFunc = &_nvml_wrapper_getClockInfo;
        event->variant = NVML_CLOCK_VIDEO;
        event++;
    }

    return 0;
}


static int
_nvml_getFeaturesOfDevice(NvmlDevice* device)
{
    int num_events = 0;
    unsigned int value;

    if ((*nvmlDeviceGetClockInfo_ptr)(device->nvmlDevice, NVML_CLOCK_GRAPHICS, &value) == NVML_SUCCESS)
    {
        device->features |= FEATURE_CLOCK_INFO;
        num_events += 4;
    }

    device->numAllEvents = num_events;
    return 0;
}


static int
_nvml_createDevice(int idx, NvmlDevice* device)
{
    int ret;

    // Set corresponding nvmon device
    device->nvDevice = &nvGroupSet->gpus[idx];
    device->features = 0;
    device->activeEventSet = 0;
    device->numEventSets = 0;
    device->eventSets = NULL;

    // Get NVML device handle
    NVML_CALL(nvmlDeviceGetHandleByIndex_v2, (device->nvDevice->deviceId, &device->nvmlDevice), {
        ERROR_PRINT(Failed to get device handle for GPU %d, device->nvDevice->deviceId);
        return -1;
    });

    ret = _nvml_getFeaturesOfDevice(device);
    if (ret < 0) return ret;

    // Allocate memory for event list
    device->allEvents = (NvmlEvent*) malloc(device->numAllEvents * sizeof(NvmlEvent));
    if (device->allEvents == NULL)
    {
        ERROR_PRINT(Failed to allocate memory for event list of GPU %d, device->nvDevice->deviceId);
        return -ENOMEM;
    }

    ret = _nvml_getEventsForDevice(device);
    if (ret < 0) return ret;

    return 0;
}


static int
_nvml_readCounters(void (*saveTimestamp)(NvmlDevice* device, uint64_t timestamp), void (*afterMeasure)(NvmlEventResult* result))
{
    int ret;

    // Get timestamp
    uint64_t timestamp;
    CUPTI_CALL(cuptiGetTimestamp, (&timestamp), return -EFAULT);
    if (ret < 0)
    {
        return -EFAULT;
    }

    for (int i = 0; i < nvmlContext.numDevices; i++)
    {
        NvmlDevice* device = &nvmlContext.devices[i];
        NvmlEventSet* eventSet = &device->eventSets[device->activeEventSet];

        // Save timestamp
        if (saveTimestamp)
        {
            saveTimestamp(device, timestamp);
        }

        // Read value of each event
        for (int i = 0; i < eventSet->numEvents; i++)
        {
            NvmlEvent* event = &eventSet->events[i];
            NvmlEventResult* result = &eventSet->results[i];
            if (event->measureFunc)
            {
                ret = event->measureFunc(device->nvmlDevice, event, result);
                if (ret < 0) return ret;

                if (afterMeasure)
                {
                    afterMeasure(result);
                }
            }
        }
    }

    return 0;
}


static void
_nvml_saveStartTime(NvmlDevice* device, uint64_t timestamp)
{
    device->time.start = timestamp;
    device->time.read = timestamp;
}


static void
_nvml_resetFullValue(NvmlEventResult* result)
{
    result->fullValue = 0;
}


static void
_nvml_saveReadTime(NvmlDevice* device, uint64_t timestamp)
{
    device->time.read = timestamp;
}


static void
_nvml_saveStopTime(NvmlDevice* device, uint64_t timestamp)
{
    device->time.stop = timestamp;
}


// ----------------------------------------------------
//   Exported functions
// ----------------------------------------------------

int
nvml_init()
{
    int ret;

    if (nvml_initialized == 1)
    {
        return 0;
    }

    ret = _nvml_linkLibraries();
    if (ret < 0)
    {
        ERROR_PLAIN_PRINT(Failed to link libraries);
        return -1;
    }

    // Allocate space for nvml specific structures
    nvmlContext.numDevices = nvGroupSet->numberOfGPUs;
    nvmlContext.devices = (NvmlDevice*) malloc(nvmlContext.numDevices * sizeof(NvmlDevice));
    if (nvmlContext.devices == NULL)
    {   
        ERROR_PLAIN_PRINT(Cannot allocate NVML device structures);
        return -ENOMEM;
    }

    // Init NVML
    NVML_CALL(nvmlInit_v2, (), return -1);

    // Do device specific setup
    for (int i = 0; i < nvmlContext.numDevices; i++)
    {
        NvmlDevice* device = &nvmlContext.devices[i];
        ret = _nvml_createDevice(i, device);
        if (ret < 0)
        {
            ERROR_PRINT(Failed to create device #%d, i);
            return ret;
        }
    }

    nvml_initialized = 1;
    return 0;
}


void
nvml_finalize()
{
    if (nvmlContext.devices)
    {
        for (int i = 0; i < nvmlContext.numDevices; i++)
        {
            NvmlDevice* device = &nvmlContext.devices[i];

            FREE_IF_NOT_NULL(device->allEvents);
            for (int j = 0; j < device->numEventSets; j++)
            {
                FREE_IF_NOT_NULL(device->eventSets[j].events);
                FREE_IF_NOT_NULL(device->eventSets[j].results);
            }
            FREE_IF_NOT_NULL(device->eventSets);
        }
        free(nvmlContext.devices);
    }

    // Shutdown NVML
    NVML_CALL(nvmlShutdown, (), return);
}


int
nvml_addEventSet(char** events, int numEvents)
{
    // Allocate memory for event results
    for (int i = 0; i < nvmlContext.numDevices; i++)
    {
        NvmlDevice* device = &nvmlContext.devices[i];

        // Allocate new event set in device
        NvmlEvent* tmpEvents = (NvmlEvent*) malloc(numEvents * sizeof(NvmlEvent));
        if (tmpEvents == NULL)
        {
            ERROR_PLAIN_PRINT(Cannot allocate events for new event set);
            return -ENOMEM;
        }
        NvmlEventResult* tmpResults = (NvmlEventResult*) malloc(numEvents * sizeof(NvmlEventResult));
        if (tmpResults == NULL)
        {
            ERROR_PLAIN_PRINT(Cannot allocate event results);
            free(tmpEvents);
            return -ENOMEM;
        }
        NvmlEventSet* tmpEventSets = (NvmlEventSet*) realloc(device->eventSets, (device->numEventSets+1) * sizeof(NvmlEventSet));
        if (tmpEventSets == NULL)
        {
            ERROR_PLAIN_PRINT(Cannot allocate new event set);
            free(tmpEvents);
            free(tmpResults);
            return -ENOMEM;
        }

        // Copy event information
        for (int j = 0; j < numEvents; j++)
        {
            // Search for it in allEvents
            int idx = -1;
            for (int k = 0; k < device->numAllEvents; k++)
            {
                if (strcmp(device->allEvents[k].name, events[j]) == 0)
                {
                    idx = k;
                    break;
                }
            }

            // Check if event was found
            if (idx < 0)
            {
                ERROR_PRINT(Could not find event %s, events[j]);
                return -EINVAL;
            }

            // Copy whole event into activeEvents array
            memcpy(&tmpEvents[j], &device->allEvents[idx], sizeof(NvmlEvent));
        }

        device->eventSets = tmpEventSets;
        device->eventSets[device->numEventSets].numEvents = numEvents;
        device->eventSets[device->numEventSets].events = tmpEvents;
        device->eventSets[device->numEventSets].results = tmpResults;
        device->numEventSets++;
    }

    return 0;
}


int
nvml_setupCounters(int gid)
{
    // Update active events of each device
    for (int i = 0; i < nvmlContext.numDevices; i++)
    {
        nvmlContext.devices[i].activeEventSet = gid;
    }

    return 0;
}


int
nvml_getEventsOfGpu(int gpuId, NvmonEventList_t* output)
{
    int gpuIdx = -1;

    // Find index with given gpuId
    for (int i = 0; i < nvmlContext.numDevices; i++)
    {
        if (nvmlContext.devices[i].nvDevice->deviceId == gpuId)
        {
            gpuIdx = i;
            break;
        }
    }
    if (gpuIdx < 0)
    {
        return -EINVAL;
    }

    // Get device handle
    NvmlDevice* device = &nvmlContext.devices[gpuIdx];

    // Allocate space for output structure
    NvmonEventListEntry* entries = (NvmonEventListEntry*) malloc(device->numAllEvents * sizeof(NvmonEventListEntry));
    if (entries == NULL)
    {
        ERROR_PLAIN_PRINT(Cannot allocate event list entries);
        return -ENOMEM;
    }
    NvmonEventList* list = (NvmonEventList*) malloc(sizeof(NvmonEventList));
    if (list == NULL)
    {
        ERROR_PLAIN_PRINT(Cannot allocate event list);
        return -ENOMEM;
    }

    // Fill structure
    for (int i = 0; i < device->numAllEvents; i++)
    {
        NvmlEvent* event = &device->allEvents[i];
        NvmonEventListEntry* entry = &entries[i];

        entry->name = event->name;
        entry->desc = "No description";
        entry->limit = "GPU";
    }

    list->events = entries;
    list->numEvents = device->numAllEvents;
    *output = list;

    return 0;
}


void
nvml_returnEventsOfGpu(NvmonEventList_t list)
{
    if (list == NULL)
    {
        return;
    }

    if (list->events != NULL && list->numEvents > 0)
    {
        // Event entries do not have owned strings, so nothing to free per entry
        free(list->events);
    }

    free(list);
}


int
nvml_startCounters()
{
    int ret;

    // Ensure nvml is initialized
    if (!nvml_initialized)
    {
        return -EFAULT;
    }

    // Read initial counter values and reset full value
    ret = _nvml_readCounters(_nvml_saveStartTime, _nvml_resetFullValue);
    if (ret < 0) return ret;

    return 0;
}


int
nvml_stopCounters()
{
    int ret;

    // Ensure nvml is initialized
    if (!nvml_initialized)
    {
        return -EFAULT;
    }

    // Read counters
    ret = _nvml_readCounters(_nvml_saveStopTime, NULL);
    if (ret < 0) return ret;

    return 0;
}


int
nvml_readCounters()
{
    int ret;

    // Ensure nvml is initialized
    if (!nvml_initialized)
    {
        return -EFAULT;
    }

    // Read counters
    ret = _nvml_readCounters(_nvml_saveReadTime, NULL);
    if (ret < 0) return ret;

    return 0;
}


int
nvml_getNumberOfEvents(int groupId)
{
    // Ensure nvml is initialized
    if (!nvml_initialized)
    {
        return -EFAULT;
    }

    // Verify that at least one device is registered
    if (nvmlContext.numDevices < 1)
    {
        return 0; // No events registered
    }

    // Verify groupId
    NvmlDevice* device = &nvmlContext.devices[0];
    if (groupId < 0 || groupId >= device->numEventSets)
    {
        return -EINVAL;
    }

    // Events are the same on all devices, take the first
    return device->eventSets[groupId].numEvents;
}


int
nvml_getResult(int gpuIdx, int groupId, int eventId)
{
    // Ensure nvml is initialized
    if (!nvml_initialized)
    {
        return -EFAULT;
    }

    // Validate gpuIdx
    if (gpuIdx < 0 || gpuIdx >= nvmlContext.numDevices)
    {
        return -EINVAL;
    }

    // Validate groupId
    NvmlDevice* device = &nvmlContext.devices[gpuIdx];
    if (groupId < 0 || groupId >= device->numEventSets)
    {
        return -EINVAL;
    }

    // Validate eventId
    NvmlEventSet* eventSet = &device->eventSets[groupId];
    if (eventId < 0 || eventId >= eventSet->numEvents)
    {
        return -EINVAL;
    }

    // Return result
    return eventSet->results[eventId].fullValue;
}


int
nvml_getLastResult(int gpuIdx, int groupId, int eventId)
{
    // Ensure nvml is initialized
    if (!nvml_initialized)
    {
        return -EFAULT;
    }

    // Validate gpuIdx
    if (gpuIdx < 0 || gpuIdx >= nvmlContext.numDevices)
    {
        return -EINVAL;
    }

    // Validate groupId
    NvmlDevice* device = &nvmlContext.devices[gpuIdx];
    if (groupId < 0 || groupId >= device->numEventSets)
    {
        return -EINVAL;
    }

    // Validate eventId
    NvmlEventSet* eventSet = &device->eventSets[groupId];
    if (eventId < 0 || eventId >= eventSet->numEvents)
    {
        return -EINVAL;
    }

    // Return result
    return eventSet->results[eventId].lastValue;
}
