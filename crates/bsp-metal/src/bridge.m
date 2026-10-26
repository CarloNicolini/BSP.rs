#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>

// Resources are retained by ARC until the synchronous Rust owner is dropped.
// Independent owners share immutable pipelines and one device command queue.
@interface BSPRuntime : NSObject
@property id<MTLDevice> device;
@property id<MTLCommandQueue> queue;
@property NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelines;
@end
@implementation BSPRuntime
@end
@interface BSPContext : NSObject
@property BSPRuntime *runtime;
@property NSMutableArray<id<MTLBuffer>> *buffers;
@property id<MTLCommandBuffer> command;
@property id<MTLComputeCommandEncoder> encoder;
@end
@implementation BSPContext
@end

void *bsp_metal_new(const char *source, char *error, size_t size) {
    @autoreleasepool {
        static BSPRuntime *runtime;
        @synchronized([BSPRuntime class]) {
            if (!runtime) {
                BSPRuntime *r = [BSPRuntime new];
                r.device = MTLCreateSystemDefaultDevice();
                if (!r.device || !r.device.hasUnifiedMemory) {
                    snprintf(error, size, "Metal requires an available unified-memory Apple GPU");
                    return NULL;
                }
                MTLCompileOptions *options = [MTLCompileOptions new];
                options.fastMathEnabled = NO;
                NSError *err = nil;
                id<MTLLibrary> library = [r.device newLibraryWithSource:@(source) options:options error:&err];
                if (!library) { snprintf(error,size,"%s",err.localizedDescription.UTF8String); return NULL; }
                r.pipelines = [NSMutableDictionary new];
                for (NSString *name in @[@"cavities", @"cavities_simd", @"cavities_wide", @"clauses3", @"clauses4", @"clauses6", @"clauses12", @"clauses_general", @"residual_blocks", @"residual_final", @"variables", @"clause_observables", @"observable_blocks", @"observable_final"]) {
                    id<MTLComputePipelineState> p = [r.device newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&err];
                    if (!p) { snprintf(error,size,"%s",err.localizedDescription.UTF8String); return NULL; }
                    if (p.threadExecutionWidth != 32 || p.maxTotalThreadsPerThreadgroup < 256) { snprintf(error,size,"unsupported Metal SIMD width or threadgroup limit"); return NULL; }
                    r.pipelines[name] = p;
                }
                r.queue = [r.device newCommandQueue];
                if (!r.queue) { snprintf(error,size,"Metal command queue allocation failed"); return NULL; }
                runtime = r;
            }
        }
        BSPContext *c = [BSPContext new]; c.runtime = runtime; c.buffers = [NSMutableArray new];
        return (__bridge_retained void *)c;
    }
}
void bsp_metal_drop(void *ctx) { @autoreleasepool { BSPContext *c = (__bridge_transfer BSPContext *)ctx; (void)c; } }
uint64_t bsp_metal_budget(void *ctx) { return ((__bridge BSPContext *)ctx).runtime.device.recommendedMaxWorkingSetSize; }
const char *bsp_metal_name(void *ctx) { return ((__bridge BSPContext *)ctx).runtime.device.name.UTF8String; }
void *bsp_metal_buffer(void *ctx, size_t length) {
    @autoreleasepool {
        BSPContext *c = (__bridge BSPContext *)ctx;
        id<MTLBuffer> b = [c.runtime.device newBufferWithLength:MAX(length,4) options:MTLResourceStorageModeShared];
        if (!b) return NULL;
        [c.buffers addObject:b]; return b.contents;
    }
}
void bsp_metal_begin(void *ctx, bool checkpoint, size_t bytes) {
    @autoreleasepool {
        BSPContext *c = (__bridge BSPContext *)ctx;
        c.command = [c.runtime.queue commandBuffer];
        if (checkpoint && bytes) {
            id<MTLBlitCommandEncoder> blit = [c.command blitCommandEncoder];
            [blit copyFromBuffer:c.buffers[8] sourceOffset:0 toBuffer:c.buffers[22] destinationOffset:0 size:bytes];
            [blit endEncoding];
        }
        c.encoder = [c.command computeCommandEncoder];
        NSUInteger offset = 0;
        for (NSUInteger i=0; i<c.buffers.count; ++i) [c.encoder setBuffer:c.buffers[i] offset:offset atIndex:i];
    }
}
void bsp_metal_dispatch(void *ctx, const char *kernel, size_t count, uint32_t base) {
    if (!count) return;
    @autoreleasepool {
        BSPContext *c = (__bridge BSPContext *)ctx;
        id<MTLComputePipelineState> p = c.runtime.pipelines[@(kernel)];
        [c.encoder setComputePipelineState:p];
        [c.encoder setBytes:&base length:sizeof(base) atIndex:26];
        // All reduction pipelines use exactly 256 threads, including their tail.
        NSUInteger groups = (count+255)/256;
        [c.encoder dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [c.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
}
bool bsp_metal_end(void *ctx, double *seconds, char *error, size_t size) {
    @autoreleasepool {
        BSPContext *c = (__bridge BSPContext *)ctx;
        [c.encoder endEncoding]; c.encoder = nil;
        [c.command commit]; [c.command waitUntilCompleted];
        *seconds = c.command.GPUEndTime-c.command.GPUStartTime;
        bool ok = c.command.status == MTLCommandBufferStatusCompleted;
        if (!ok) snprintf(error,size,"%s",c.command.error.localizedDescription.UTF8String);
        c.command = nil; return ok;
    }
}
