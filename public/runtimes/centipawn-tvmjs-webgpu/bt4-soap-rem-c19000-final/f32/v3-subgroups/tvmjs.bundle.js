(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
    typeof define === 'function' && define.amd ? define(['exports'], factory) :
    (global = typeof globalThis !== 'undefined' ? globalThis : global || self, factory(global.tvmjs = {}));
})(this, (function (exports) { 'use strict';

    /******************************************************************************
    Copyright (c) Microsoft Corporation.

    Permission to use, copy, modify, and/or distribute this software for any
    purpose with or without fee is hereby granted.

    THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
    REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
    AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
    INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
    LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
    OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
    PERFORMANCE OF THIS SOFTWARE.
    ***************************************************************************** */
    /* global Reflect, Promise, SuppressedError, Symbol, Iterator */


    function __awaiter(thisArg, _arguments, P, generator) {
        function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
        return new (P || (P = Promise))(function (resolve, reject) {
            function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
            function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
            function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
            step((generator = generator.apply(thisArg, _arguments || [])).next());
        });
    }

    typeof SuppressedError === "function" ? SuppressedError : function (error, suppressed, message) {
        var e = new Error(message);
        return e.name = "SuppressedError", e.error = error, e.suppressed = suppressed, e;
    };

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
    /**
     * Check if value is a promise type
     *
     * @param value The input value
     * @returns Whether value is promise
     */
    function isPromise(value) {
        return value !== undefined && (typeof value == "object" || typeof value == "function") && typeof value.then == "function";
    }
    /**
     * Convert string to Uint8array.
     * @param str The string.
     * @returns The corresponding Uint8Array.
     */
    function StringToUint8Array(str) {
        const arr = new TextEncoder().encode(str);
        const resArr = new Uint8Array(arr.length + 1);
        for (let i = 0; i < arr.length; ++i) {
            resArr[i] = arr[i];
        }
        resArr[arr.length] = 0;
        return resArr;
    }
    /**
     * Convert Uint8array to string.
     * @param array The array.
     * @returns The corresponding string.
     */
    function Uint8ArrayToString(arr) {
        const ret = [];
        for (const ch of arr) {
            ret.push(String.fromCharCode(ch));
        }
        return ret.join("");
    }
    /**
     * Internal assert helper
     * @param condition The condition to fail.
     * @param msg The message.
     */
    function assert(condition, msg) {
        if (!condition) {
            throw new Error("AssertError:" + (msg || ""));
        }
    }
    /**
     * Get the path to the wasm library in nodejs.
     * @return The wasm path.
     */
    function wasmPath() {
        return __dirname + "/wasm";
    }
    /**
     * Linear congruential generator for random number generating that can be seeded.
     *
     * Follows the implementation of `include/tvm/support/random_engine.h`, which follows the
     * sepcification in https://en.cppreference.com/w/cpp/numeric/random/linear_congruential_engine.
     *
     * Note `Number.MAX_SAFE_INTEGER = 2^53 - 1`, and our intermediates are strictly less than 2^48.
     */
    class LinearCongruentialGenerator {
        /**
         * Set modulus, multiplier, and increment. Initialize `rand_state` according to `Date.now()`.
         */
        constructor() {
            this.modulus = 2147483647; // 2^32 - 1
            this.multiplier = 48271; // between 2^15 and 2^16
            this.increment = 0;
            this.setSeed(Date.now());
        }
        /**
         * Sets `rand_state` after normalized with `modulus` to ensure that it is within range.
         * @param seed Any integer. Used to set `rand_state` after normalized with `modulus`.
         *
         * Postcondition: pass `checkRandState()`, i.e. rand_state > 0 and is an integer.
         */
        setSeed(seed) {
            if (!Number.isInteger(seed)) {
                throw new Error("Seed should be an integer.");
            }
            this.rand_state = seed % this.modulus;
            if (this.rand_state == 0) {
                this.rand_state = 1;
            }
            this.checkRandState();
        }
        /**
         * Generate the next integer in the range (0, this.modulus) non-inclusive, updating `rand_state`.
         *
         * Postcondition: pass `checkRandState()`, i.e. rand_state > 0 and is an integer.
         */
        nextInt() {
            // `intermediate` is always < 2^48, hence less than `Number.MAX_SAFE_INTEGER` due to the
            // invariants as commented in the constructor.
            const intermediate = this.multiplier * this.rand_state + this.increment;
            this.rand_state = intermediate % this.modulus;
            this.checkRandState();
            return this.rand_state;
        }
        /**
         * Generates random float between (0, 1) non-inclusive, updating `rand_state`.
         *
         * Postcondition: pass `checkRandState()`, i.e. rand_state > 0 and is an integer.
         */
        randomFloat() {
            return this.nextInt() / this.modulus;
        }
        checkRandState() {
            if (this.rand_state <= 0) {
                throw new Error("Random state is unexpectedly not strictly positive.");
            }
            if (!Number.isInteger(this.rand_state)) {
                throw new Error("Random state is unexpectedly not an integer.");
            }
        }
    }

    /**
     * Wasm Memory wrapper to perform JS side raw memory access.
     */
    class Memory {
        constructor(memory) {
            this.wasm32 = true;
            this.memory = memory;
            this.buffer = this.memory.buffer;
            this.viewU8 = new Uint8Array(this.buffer);
            this.viewU16 = new Uint16Array(this.buffer);
            this.viewI32 = new Int32Array(this.buffer);
            this.viewU32 = new Uint32Array(this.buffer);
            this.viewF32 = new Float32Array(this.buffer);
            this.viewF64 = new Float64Array(this.buffer);
        }
        loadU8(ptr) {
            if (this.buffer != this.memory.buffer) {
                this.updateViews();
            }
            return this.viewU8[ptr >> 0];
        }
        loadU16(ptr) {
            if (this.buffer != this.memory.buffer) {
                this.updateViews();
            }
            return this.viewU16[ptr >> 1];
        }
        loadU32(ptr) {
            if (this.buffer != this.memory.buffer) {
                this.updateViews();
            }
            return this.viewU32[ptr >> 2];
        }
        loadI32(ptr) {
            if (this.buffer != this.memory.buffer) {
                this.updateViews();
            }
            return this.viewI32[ptr >> 2];
        }
        loadI64(ptr) {
            if (this.buffer != this.memory.buffer) {
                this.updateViews();
            }
            const base = ptr >> 2;
            // assumes little endian, for now truncate high.
            return this.viewI32[base];
        }
        loadF32(ptr) {
            if (this.buffer != this.memory.buffer) {
                this.updateViews();
            }
            return this.viewF32[ptr >> 2];
        }
        loadF64(ptr) {
            if (this.buffer != this.memory.buffer) {
                this.updateViews();
            }
            return this.viewF64[ptr >> 3];
        }
        loadPointer(ptr) {
            if (this.buffer != this.memory.buffer) {
                this.updateViews();
            }
            if (this.wasm32) {
                return this.loadU32(ptr);
            }
            else {
                return this.loadI64(ptr);
            }
        }
        loadUSize(ptr) {
            if (this.buffer != this.memory.buffer) {
                this.updateViews();
            }
            if (this.wasm32) {
                return this.loadU32(ptr);
            }
            else {
                return this.loadI64(ptr);
            }
        }
        sizeofPtr() {
            return this.wasm32 ? 4 /* SizeOf.I32 */ : 8 /* SizeOf.I64 */;
        }
        /**
         * Load raw bytes from ptr.
         * @param ptr The head address
         * @param numBytes The number
         */
        loadRawBytes(ptr, numBytes) {
            if (this.buffer != this.memory.buffer) {
                this.updateViews();
            }
            const result = new Uint8Array(numBytes);
            result.set(this.viewU8.slice(ptr, ptr + numBytes));
            return result;
        }
        /**
         * Load null-terminated C-string from ptr.
         * @param ptr The head address
         */
        loadCString(ptr) {
            if (this.buffer != this.memory.buffer) {
                this.updateViews();
            }
            // NOTE: the views are still valid for read.
            const ret = [];
            let ch = 1;
            while (ch != 0) {
                ch = this.viewU8[ptr];
                if (ch != 0) {
                    ret.push(String.fromCharCode(ch));
                }
                ++ptr;
            }
            return ret.join("");
        }
        /**
         * Store raw bytes to the ptr.
         * @param ptr The head address.
         * @param bytes The bytes content.
         */
        storeRawBytes(ptr, bytes) {
            if (this.buffer != this.memory.buffer) {
                this.updateViews();
            }
            this.viewU8.set(bytes, ptr);
        }
        // the following functions are related to TVM FFI
        /**
         * Load the object type index from the object handle.
         * @param objectHandle The handle of the object.
         * @returns The object type index.
         */
        loadObjectTypeIndex(objectHandle) {
            // The object layout is [ref_counter (i64), type_index (i32), ...].
            return this.loadI32(objectHandle + 8 /* SizeOf.I64 */);
        }
        /**
         * Load the type key from the type info pointer.
         * @param typeInfoPtr The pointer to the type info.
         * @returns The type key.
         */
        loadTypeInfoTypeKey(typeInfoPtr) {
            const typeKeyPtr = typeInfoPtr + 2 * 4 /* SizeOf.I32 */;
            return this.loadByteArrayAsString(typeKeyPtr);
        }
        /**
         * Load small string from value pointer.
         * @param ffiAnyPtr The pointer to the value.
         * @returns The small string.
         */
        loadSmallStr(ffiAnyPtr) {
            if (this.buffer != this.memory.buffer) {
                this.updateViews();
            }
            const sizePtr = ffiAnyPtr + 4 /* SizeOf.I32 */;
            const length = this.loadU32(sizePtr);
            const dataPtr = ffiAnyPtr + 4 /* SizeOf.I32 */ + 4 /* SizeOf.I32 */;
            const ret = [];
            for (let i = 0; i < length; i++) {
                ret.push(String.fromCharCode(this.viewU8[dataPtr + i]));
            }
            return ret.join("");
        }
        /**
         * Load small bytes from value pointer.
         * @param ffiAnyPtr
         */
        loadSmallBytes(ffiAnyPtr) {
            if (this.buffer != this.memory.buffer) {
                this.updateViews();
            }
            const sizePtr = ffiAnyPtr + 4 /* SizeOf.I32 */;
            const length = this.loadU32(sizePtr);
            const dataPtr = ffiAnyPtr + 4 /* SizeOf.I32 */ + 4 /* SizeOf.I32 */;
            const result = new Uint8Array(length);
            result.set(this.viewU8.slice(dataPtr, dataPtr + length));
            return result;
        }
        /**
         * Load bytearray as string from ptr.
         * @param byteArrayPtr The head address of the bytearray.
         */
        loadByteArrayAsString(byteArrayPtr) {
            if (this.buffer != this.memory.buffer) {
                this.updateViews();
            }
            const ptr = this.loadPointer(byteArrayPtr);
            const length = this.loadUSize(byteArrayPtr + this.sizeofPtr());
            // NOTE: the views are still valid for read.
            const ret = [];
            for (let i = 0; i < length; i++) {
                ret.push(String.fromCharCode(this.viewU8[ptr + i]));
            }
            return ret.join("");
        }
        /**
         * Load bytearray as bytes from ptr.
         * @param byteArrayPtr The head address of the bytearray.
         */
        loadByteArrayAsBytes(byteArrayPtr) {
            if (this.buffer != this.memory.buffer) {
                this.updateViews();
            }
            const ptr = this.loadPointer(byteArrayPtr);
            const length = this.loadUSize(byteArrayPtr + this.sizeofPtr());
            const result = new Uint8Array(length);
            result.set(this.viewU8.slice(ptr, ptr + length));
            return result;
        }
        // private functions
        /**
         * Update memory view after the memory growth.
         */
        updateViews() {
            this.buffer = this.memory.buffer;
            this.viewU8 = new Uint8Array(this.buffer);
            this.viewU16 = new Uint16Array(this.buffer);
            this.viewI32 = new Int32Array(this.buffer);
            this.viewU32 = new Uint32Array(this.buffer);
            this.viewF32 = new Float32Array(this.buffer);
            this.viewF64 = new Float64Array(this.buffer);
        }
    }
    /**
     * Auxiliary call stack for the FFI calls.
     *
     * Lifecyle of a call stack.
     * - Calls into allocXX to allocate space, mixed with storeXXX to store data.
     * - Calls into ptrFromOffset, no further allocation(as ptrFromOffset can change),
     *   can still call into storeXX
     * - Calls into commitToWasmMemory once.
     * - reset.
     */
    class CachedCallStack {
        constructor(memory, allocSpace, freeSpace) {
            /** List of temporay arguments that can be disposed during reset. */
            this.tempArgs = [];
            this.stackTop = 0;
            this.basePtr = 0;
            this.addressToSetTargetValue = [];
            const initCallStackSize = 128;
            this.memory = memory;
            this.cAllocSpace = allocSpace;
            this.cFreeSpace = freeSpace;
            this.buffer = new ArrayBuffer(initCallStackSize);
            this.basePtr = this.cAllocSpace(initCallStackSize);
            this.viewU8 = new Uint8Array(this.buffer);
            this.viewI32 = new Int32Array(this.buffer);
            this.viewU32 = new Uint32Array(this.buffer);
            this.viewF64 = new Float64Array(this.buffer);
            this.updateViews();
        }
        dispose() {
            if (this.basePtr != 0) {
                this.cFreeSpace(this.basePtr);
                this.basePtr = 0;
            }
        }
        /**
         * Rest the call stack so that it can be reused again.
         */
        reset() {
            this.stackTop = 0;
            assert(this.addressToSetTargetValue.length === 0);
            while (this.tempArgs.length != 0) {
                this.tempArgs.pop().dispose();
            }
        }
        /**
         * Commit all the cached data to WasmMemory.
         * This function can only be called once.
         * No further store function should be called.
         *
         * @param nbytes Number of bytes to be stored.
         */
        commitToWasmMemory(nbytes = this.stackTop) {
            // commit all pointer values.
            while (this.addressToSetTargetValue.length != 0) {
                const [targetOffset, valueOffset] = this.addressToSetTargetValue.pop();
                this.storePtr(targetOffset, this.ptrFromOffset(valueOffset));
            }
            this.memory.storeRawBytes(this.basePtr, this.viewU8.slice(0, nbytes));
        }
        /**
         * Allocate space by number of bytes
         * @param nbytes Number of bytes.
          * Note: This function always allocate space that aligns to 64bit.
         */
        allocRawBytes(nbytes) {
            // always aligns to 64bit
            nbytes = ((nbytes + 7) >> 3) << 3;
            if (this.stackTop + nbytes > this.buffer.byteLength) {
                const newSize = Math.max(this.buffer.byteLength * 2, this.stackTop + nbytes);
                const oldU8 = this.viewU8;
                this.buffer = new ArrayBuffer(newSize);
                this.updateViews();
                this.viewU8.set(oldU8);
                if (this.basePtr != 0) {
                    this.cFreeSpace(this.basePtr);
                }
                this.basePtr = this.cAllocSpace(newSize);
            }
            const retOffset = this.stackTop;
            this.stackTop += nbytes;
            return retOffset;
        }
        /**
         * Allocate space for pointers.
         * @param count Number of pointers.
         * @returns The allocated pointer array.
         */
        allocPtrArray(count) {
            return this.allocRawBytes(this.memory.sizeofPtr() * count);
        }
        /**
         * Get the real pointer from offset values.
         * Note that the returned value becomes obsolete if alloc is called on the stack.
         * @param offset The allocated offset.
         */
        ptrFromOffset(offset) {
            return this.basePtr + offset;
        }
        // Store APIs
        storePtr(offset, value) {
            if (this.memory.wasm32) {
                this.storeU32(offset, value);
            }
            else {
                this.storeI64(offset, value);
            }
        }
        storeUSize(offset, value) {
            if (this.memory.wasm32) {
                this.storeU32(offset, value);
            }
            else {
                this.storeI64(offset, value);
            }
        }
        storeI32(offset, value) {
            this.viewI32[offset >> 2] = value;
        }
        storeU32(offset, value) {
            this.viewU32[offset >> 2] = value;
        }
        storeI64(offset, value) {
            // For now, just store as 32bit
            // NOTE: wasm always uses little endian.
            const low = value & 0xffffffff;
            const base = offset >> 2;
            this.viewI32[base] = low;
            // sign extend
            this.viewI32[base + 1] = value < 0 ? -1 : 0;
        }
        storeF64(offset, value) {
            this.viewF64[offset >> 3] = value;
        }
        storeRawBytes(offset, bytes) {
            this.viewU8.set(bytes, offset);
        }
        /**
         * Allocate a byte array for a string and return the offset of the byte array.
         * @param data The string to allocate.
         * @returns The offset of the byte array.
         */
        allocByteArrayForString(data) {
            const dataUint8 = StringToUint8Array(data);
            // Note: size of size_t equals sizeof ptr.
            const headerOffset = this.allocRawBytes(this.memory.sizeofPtr() * 2);
            const dataOffset = this.allocRawBytes(dataUint8.length);
            this.storeUSize(headerOffset + this.memory.sizeofPtr(), data.length);
            this.storeRawBytes(dataOffset, dataUint8);
            this.addressToSetTargetValue.push([headerOffset, dataOffset]);
            return headerOffset;
        }
        /**
         * Allocate then set C-String pointer to the offset.
         * This function will call into allocBytes to allocate necessary data.
         * The address won't be set immediately(because the possible change of basePtr)
         * and will be filled when we commit the data.
         *
         * @param offset The offset to set ot data pointer.
         * @param data The string content.
         */
        allocThenSetArgString(offset, data) {
            const dataUint8 = StringToUint8Array(data);
            const strOffset = this.allocRawBytes(dataUint8.length);
            this.storeRawBytes(strOffset, dataUint8);
            this.addressToSetTargetValue.push([offset, strOffset]);
        }
        /**
         * Allocate then set the argument location with a TVMByteArray.
         * Allocate new temporary space for bytes.
         *
         * @param offset The offset to set ot data pointer.
         * @param data The string content.
         */
        allocThenSetArgBytes(offset, data) {
            // Note: size of size_t equals sizeof ptr.
            const headerOffset = this.allocRawBytes(this.memory.sizeofPtr() * 2);
            const dataOffset = this.allocRawBytes(data.length);
            this.storeRawBytes(dataOffset, data);
            this.storeUSize(headerOffset + this.memory.sizeofPtr(), data.length);
            this.addressToSetTargetValue.push([offset, headerOffset]);
            this.addressToSetTargetValue.push([headerOffset, dataOffset]);
        }
        /**
         * Update internal cache views.
         */
        updateViews() {
            this.viewU8 = new Uint8Array(this.buffer);
            this.viewI32 = new Int32Array(this.buffer);
            this.viewU32 = new Uint32Array(this.buffer);
            this.viewF64 = new Float64Array(this.buffer);
        }
    }

    /**
     * Detect library provider from the importObject.
     *
     * @param importObject The import object.
     */
    function detectLibraryProvider(importObject) {
        if (importObject["wasmLibraryProvider"] &&
            importObject["wasmLibraryProvider"]["start"] &&
            importObject["wasmLibraryProvider"]["imports"] !== undefined) {
            const item = importObject;
            // create provider so that we capture imports in the provider.
            return {
                imports: item.wasmLibraryProvider.imports,
                start: (inst) => {
                    item.wasmLibraryProvider.start(inst);
                },
            };
        }
        else if (importObject["imports"] && importObject["start"] !== undefined) {
            return importObject;
        }
        else if (importObject["wasiImport"] && importObject["start"] !== undefined) {
            // WASI
            return {
                imports: {
                    "wasi_snapshot_preview1": importObject["wasiImport"],
                },
                start: (inst) => {
                    importObject["start"](inst);
                }
            };
        }
        else {
            return undefined;
        }
    }
    /**
     * Environment to impelement most of the JS library functions.
     */
    class Environment {
        constructor(importObject = {}, logger = console.log) {
            /**
             * Maintains a table of FTVMWasmPackedCFunc that the C part
             * can call via TVMWasmPackedCFunc.
             *
             * We maintain a separate table so that we can have un-limited amount
             * of functions that do not maps to the address space.
             */
            this.packedCFuncTable = [
                undefined,
            ];
            /**
             * Free table index that can be recycled.
             */
            this.packedCFuncTableFreeId = [];
            this.logger = logger;
            this.libProvider = detectLibraryProvider(importObject);
            // get imports from the provider
            if (this.libProvider !== undefined) {
                this.imports = this.libProvider.imports;
            }
            else {
                this.imports = importObject;
            }
            // update with more functions
            this.imports.env = this.environment(this.imports.env);
        }
        /** Mark the start of the instance. */
        start(inst) {
            if (this.libProvider !== undefined) {
                this.libProvider.start(inst);
            }
        }
        environment(initEnv) {
            // default env can be overriden by libraries.
            const defaultEnv = {
                "__cxa_thread_atexit": () => { },
                "emscripten_notify_memory_growth": (index) => { }
            };
            const wasmSafeCall = (self, args, num_args, result) => {
                const cfunc = this.packedCFuncTable[self];
                assert(cfunc !== undefined);
                return cfunc(self, args, num_args, result);
            };
            const wasmFunctionDeleter = (self) => {
                this.packedCFuncTable[self] = undefined;
                this.packedCFuncTableFreeId.push(self);
            };
            const newEnv = {
                "TVMFFIWasmSafeCall": wasmSafeCall,
                "TVMFFIWasmFunctionDeleter": wasmFunctionDeleter,
                "__console_log": (msg) => {
                    this.logger(msg);
                }
            };
            return Object.assign(defaultEnv, initEnv, newEnv);
        }
    }

    /** The start location of asynctify stack data */
    const ASYNCIFY_DATA_ADDR = 16;
    /** The data start of stack rewind/unwind */
    const ASYNCIFY_DATA_START = ASYNCIFY_DATA_ADDR + 8;
    /** The data end of stack rewind/unwind */
    const ASYNCIFY_DATA_END = 1024;
    /** Hold asynctify handler instance that runtime can use */
    class AsyncifyHandler {
        constructor(exports, memory) {
            /** current state kind */
            this.state = 0 /* AsyncifyStateKind.None */;
            /** The stored value before unwind */
            this.storedPromiseBeforeUnwind = null;
            // NOTE: asynctify do not work with exceptions
            // this implementation here is mainly for possible future compact
            /** The stored value that is resolved */
            this.storedValueBeforeRewind = null;
            /** The stored exception */
            this.storedExceptionBeforeRewind = null;
            this.exports = exports;
            this.initMemory(memory);
        }
        // NOTE: wrapImport and wrapExport are closely related to each other
        // We mark the logical jump pt in comments to increase the readability
        /**
         * Whether the wasm enables asynctify
         * @returns Whether the wasm enables asynctify
         */
        enabled() {
            return this.exports.asyncify_stop_rewind !== undefined;
        }
        /**
         * Get the current asynctify state
         *
         * @returns The current asynctify state
         */
        isNormalStackState() {
            return this.state == 0 /* AsyncifyStateKind.None */;
        }
        /**
         * Get the current asynctify state
         *
         * @returns The current asynctify state
         */
        getState() {
            return this.state;
        }
        /**
         * Wrap a function that can be used as import of the wasm asynctify layer
         *
         * @param func The input import function
         * @returns The wrapped function that can be registered to the system
         */
        wrapImport(func) {
            return (...args) => {
                // this is being called second time
                // where we are rewinding the stack
                if (this.getState() == 2 /* AsyncifyStateKind.Rewinding */) {
                    // JUMP-PT-REWIND: rewind will jump to this pt
                    // while rewinding the stack
                    this.stopRewind();
                    // the value has been resolved
                    if (this.storedValueBeforeRewind !== null) {
                        assert(this.storedExceptionBeforeRewind === null);
                        const result = this.storedValueBeforeRewind;
                        this.storedValueBeforeRewind = null;
                        return result;
                    }
                    else {
                        assert(this.storedValueBeforeRewind === null);
                        const error = this.storedExceptionBeforeRewind;
                        this.storedExceptionBeforeRewind = null;
                        throw error;
                    }
                }
                // this function is being called for the first time
                assert(this.getState() == 0 /* AsyncifyStateKind.None */);
                // call the function
                const value = func(...args);
                // if the value is promise
                // we need to unwind the stack
                // so the caller will be able to evaluate the promise
                if (isPromise(value)) {
                    // The next code step is JUMP-PT-UNWIND in wrapExport
                    // The value will be passed to that pt through storedPromiseBeforeUnwind
                    // getState() == Unwinding and we will enter the while loop in wrapExport
                    this.startUnwind();
                    assert(this.storedPromiseBeforeUnwind == null);
                    this.storedPromiseBeforeUnwind = value;
                    return undefined;
                }
                else {
                    // The next code step is JUMP-PT-UNWIND in wrapExport
                    // normal value, we don't have to do anything
                    // getState() == None and we will exit while loop there
                    return value;
                }
            };
        }
        /**
         * Warp an exported asynctify function so it can return promise
         *
         * @param func The input function
         * @returns The wrapped async function
         */
        wrapExport(func) {
            return (...args) => __awaiter(this, void 0, void 0, function* () {
                assert(this.getState() == 0 /* AsyncifyStateKind.None */);
                // call the original function
                let result = func(...args);
                // JUMP-PT-UNWIND
                // after calling the function
                // the caller may hit a unwinding point depending on
                // the if (isPromise(value)) condition in wrapImport
                while (this.getState() == 1 /* AsyncifyStateKind.Unwinding */) {
                    this.stopUnwind();
                    // try to resolve the promise that the internal requested
                    // we then store it into the temp value in storedValueBeforeRewind
                    // which then get passed onto the function(see wrapImport)
                    // that can return the value
                    const storedPromiseBeforeUnwind = this.storedPromiseBeforeUnwind;
                    this.storedPromiseBeforeUnwind = null;
                    assert(this.storedExceptionBeforeRewind === null);
                    assert(this.storedValueBeforeRewind == null);
                    try {
                        this.storedValueBeforeRewind = yield storedPromiseBeforeUnwind;
                    }
                    catch (error) {
                        // the store exception
                        this.storedExceptionBeforeRewind = error;
                    }
                    assert(!isPromise(this.storedValueBeforeRewind));
                    // because we called asynctify_stop_unwind,the state is now none
                    assert(this.getState() == 0 /* AsyncifyStateKind.None */);
                    // re-enter the function, jump to JUMP-PT-REWIND in wrapImport
                    // the value will be passed to that point via storedValueBeforeRewind
                    //
                    // NOTE: we guarantee that if exception is throw the asynctify state
                    // will already be at None, this is because we will goto JUMP-PT-REWIND
                    // which will call aynctify_stop_rewind
                    this.startRewind();
                    result = func(...args);
                }
                return result;
            });
        }
        startRewind() {
            if (this.exports.asyncify_start_rewind === undefined) {
                throw Error("Asynctify is not enabled, please compile with -s ASYNCIFY=1 in emcc");
            }
            this.exports.asyncify_start_rewind(ASYNCIFY_DATA_ADDR);
            this.state = 2 /* AsyncifyStateKind.Rewinding */;
        }
        stopRewind() {
            if (this.exports.asyncify_stop_rewind === undefined) {
                throw Error("Asynctify is not enabled, please compile with -s ASYNCIFY=1 in emcc");
            }
            this.exports.asyncify_stop_rewind();
            this.state = 0 /* AsyncifyStateKind.None */;
        }
        startUnwind() {
            if (this.exports.asyncify_start_unwind === undefined) {
                throw Error("Asynctify is not enabled, please compile with -s ASYNCIFY=1 in emcc");
            }
            this.exports.asyncify_start_unwind(ASYNCIFY_DATA_ADDR);
            this.state = 1 /* AsyncifyStateKind.Unwinding */;
        }
        stopUnwind() {
            if (this.exports.asyncify_stop_unwind === undefined) {
                throw Error("Asynctify is not enabled, please compile with -s ASYNCIFY=1 in emcc");
            }
            this.exports.asyncify_stop_unwind();
            this.state = 0 /* AsyncifyStateKind.None */;
        }
        /**
         * Initialize the wasm memory to setup necessary meta-data
         * for asynctify handling
         * @param memory The memory ti
         */
        initMemory(memory) {
            // Set the meta-data at address ASYNCTIFY_DATA_ADDR
            new Int32Array(memory.buffer, ASYNCIFY_DATA_ADDR, 2).set([ASYNCIFY_DATA_START, ASYNCIFY_DATA_END]);
        }
    }

    /**
     * DetectGPU device in the environment.
     */
    function detectGPUDevice() {
        return __awaiter(this, arguments, void 0, function* (powerPreference = "high-performance") {
            if (typeof navigator !== "undefined" && navigator.gpu !== undefined) {
                const adapter = yield navigator.gpu.requestAdapter({ powerPreference });
                if (adapter == null) {
                    throw Error("Unable to find a compatible GPU. This issue might be because your computer doesn't have a GPU, or your system settings are not configured properly. " +
                        "Please check if your device has a GPU properly set up and if your your browser supports WebGPU. " +
                        "You can also consult your browser's compatibility chart to see if it supports WebGPU. " +
                        "For more information about WebGPU support in your browser, visit https://webgpureport.org/");
                }
                const computeMB = (value) => {
                    return Math.ceil(value / (1 << 20)) + "MB";
                };
                // more detailed error message
                let requiredMaxBufferSize = 1 << 30; // 1GB
                if (requiredMaxBufferSize > adapter.limits.maxBufferSize) {
                    // If 1GB is too large, try 256MB (default size stated in WebGPU doc)
                    const backupRequiredMaxBufferSize = 1 << 28; // 256MB
                    console.log(`Requested maxBufferSize exceeds limit. \n` +
                        `requested=${computeMB(requiredMaxBufferSize)}, \n` +
                        `limit=${computeMB(adapter.limits.maxBufferSize)}. \n` +
                        `WARNING: Falling back to ${computeMB(backupRequiredMaxBufferSize)}...`);
                    requiredMaxBufferSize = backupRequiredMaxBufferSize;
                    if (backupRequiredMaxBufferSize > adapter.limits.maxBufferSize) {
                        // Fail if 256MB is still too big
                        throw Error(`Cannot initialize runtime because of requested maxBufferSize ` +
                            `exceeds limit. requested=${computeMB(backupRequiredMaxBufferSize)}, ` +
                            `limit=${computeMB(adapter.limits.maxBufferSize)}. ` +
                            `Consider upgrading your browser.`);
                    }
                }
                let requiredMaxStorageBufferBindingSize = 1 << 30; // 1GB
                if (requiredMaxStorageBufferBindingSize > adapter.limits.maxStorageBufferBindingSize) {
                    // If 1GB is too large, try 128MB (default size stated in WebGPU doc)
                    const backupRequiredMaxStorageBufferBindingSize = 1 << 27; // 128MB
                    console.log(`Requested maxStorageBufferBindingSize exceeds limit. \n` +
                        `requested=${computeMB(requiredMaxStorageBufferBindingSize)}, \n` +
                        `limit=${computeMB(adapter.limits.maxStorageBufferBindingSize)}. \n` +
                        `WARNING: Falling back to ${computeMB(backupRequiredMaxStorageBufferBindingSize)}...`);
                    requiredMaxStorageBufferBindingSize = backupRequiredMaxStorageBufferBindingSize;
                    if (backupRequiredMaxStorageBufferBindingSize > adapter.limits.maxStorageBufferBindingSize) {
                        // Fail if 128MB is still too big
                        throw Error(`Cannot initialize runtime because of requested maxStorageBufferBindingSize ` +
                            `exceeds limit. requested=${computeMB(backupRequiredMaxStorageBufferBindingSize)}, ` +
                            `limit=${computeMB(adapter.limits.maxStorageBufferBindingSize)}. `);
                    }
                }
                const requiredMaxComputeWorkgroupStorageSize = 32 << 10;
                if (requiredMaxComputeWorkgroupStorageSize > adapter.limits.maxComputeWorkgroupStorageSize) {
                    throw Error(`Cannot initialize runtime because of requested maxComputeWorkgroupStorageSize ` +
                        `exceeds limit. requested=${requiredMaxComputeWorkgroupStorageSize}, ` +
                        `limit=${adapter.limits.maxComputeWorkgroupStorageSize}. `);
                }
                const requiredMaxStorageBuffersPerShaderStage = 10; // default is 8
                if (requiredMaxStorageBuffersPerShaderStage > adapter.limits.maxStorageBuffersPerShaderStage) {
                    throw Error(`Cannot initialize runtime because of requested maxStorageBuffersPerShaderStage ` +
                        `exceeds limit. requested=${requiredMaxStorageBuffersPerShaderStage}, ` +
                        `limit=${adapter.limits.maxStorageBuffersPerShaderStage}. `);
                }
                const candidates = [1024, 512, 256];
                const limit = adapter.limits.maxComputeInvocationsPerWorkgroup;
                const requiredMaxComputeInvocationsPerWorkgroup = candidates.find(x => x <= limit) || undefined;
                if (requiredMaxComputeInvocationsPerWorkgroup === undefined) {
                    console.log(`No candidate fits device limit=${limit}; will rely on defaults`);
                }
                else if (requiredMaxComputeInvocationsPerWorkgroup !== 1024) {
                    console.log(`Falling back to maxComputeInvocationsPerWorkgroup=${requiredMaxComputeInvocationsPerWorkgroup} ` +
                        `due to device limit=${limit}`);
                }
                const requiredFeatures = [];
                // Always require f16 if available
                if (adapter.features.has("shader-f16")) {
                    requiredFeatures.push("shader-f16");
                }
                if (adapter.features.has("subgroups")) {
                    requiredFeatures.push("subgroups");
                }
                // requestAdapterInfo() is deprecated, causing requestAdapterInfo to raise
                // issue when building. However, it is still needed for older browsers, hence `as any`.
                const adapterInfo = adapter.info || (yield adapter.requestAdapterInfo());
                const device = yield adapter.requestDevice({
                    requiredLimits: {
                        maxBufferSize: requiredMaxBufferSize,
                        maxStorageBufferBindingSize: requiredMaxStorageBufferBindingSize,
                        maxComputeWorkgroupStorageSize: requiredMaxComputeWorkgroupStorageSize,
                        maxStorageBuffersPerShaderStage: requiredMaxStorageBuffersPerShaderStage,
                        maxComputeInvocationsPerWorkgroup: requiredMaxComputeInvocationsPerWorkgroup,
                    },
                    requiredFeatures
                });
                return {
                    adapter: adapter,
                    adapterInfo: adapterInfo,
                    device: device
                };
            }
            else {
                return undefined;
            }
        });
    }
    /**
     * Create GPU buffer with `createBuffer()` but with error catching; destroy if error caught.
     * @param device The GPUDevice used to create a buffer.
     * @param descriptor The GPUBufferDescriptor passed to `createBuffer()`.
     * @returns The buffer created by `createBuffer()`.
     *
     * Note: We treat any error occurred at `createBuffer()` fatal and expect the user to handle
     *   `device.destroy()` with `device.lost.then()`.
     */
    function tryCreateBuffer(device, descriptor) {
        device.pushErrorScope("out-of-memory");
        device.pushErrorScope("validation");
        device.pushErrorScope("internal");
        const buffer = device.createBuffer(descriptor);
        device.popErrorScope().then((error) => { if (error) {
            device.destroy();
            console.error(error);
        } });
        device.popErrorScope().then((error) => { if (error) {
            device.destroy();
            console.error(error);
        } });
        device.popErrorScope().then((error) => { if (error) {
            device.destroy();
            console.error(error);
        } });
        return buffer;
    }
    const canvasRenderWGSL = `
@group(0) @binding(0) var my_sampler : sampler;
@group(0) @binding(1) var my_texture : texture_2d<f32>;

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) uv : vec2<f32>,
}

@vertex
fn vertex_main(@builtin(vertex_index) vidx : u32) -> VertexOutput {
  const pos = array(
    vec2( 1.0,  1.0),
    vec2( 1.0, -1.0),
    vec2(-1.0, -1.0),
    vec2( 1.0,  1.0),
    vec2(-1.0, -1.0),
    vec2(-1.0,  1.0),
  );

  const uv = array(
    vec2(1.0, 0.0),
    vec2(1.0, 1.0),
    vec2(0.0, 1.0),
    vec2(1.0, 0.0),
    vec2(0.0, 1.0),
    vec2(0.0, 0.0),
  );

  var output : VertexOutput;
  output.position = vec4(pos[vidx], 0.0, 1.0);
  output.uv = uv[vidx];
  return output;
}

@fragment
fn fragment_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
  return textureSample(my_texture, my_sampler, uv);
}

@fragment
fn fragment_clear(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
  return vec4(1.0, 1.0, 1.0, 1.0);
}
`;
    class CanvasRenderManager {
        constructor(device, canvas) {
            this.device = device;
            const ctx = canvas.getContext("webgpu");
            if (ctx == null) {
                throw Error("Cannot bind WebGPU context");
            }
            // avoid possible ts complain
            this.canvasContext = ctx;
            this.canvasTextureFormat = navigator.gpu.getPreferredCanvasFormat();
            this.canvasContext.configure({
                device: this.device,
                format: this.canvasTextureFormat,
                alphaMode: "opaque",
            });
            this.renderPipeline = device.createRenderPipeline({
                layout: "auto",
                vertex: {
                    module: device.createShaderModule({
                        code: canvasRenderWGSL,
                    }),
                    entryPoint: "vertex_main",
                },
                fragment: {
                    module: device.createShaderModule({
                        code: canvasRenderWGSL,
                    }),
                    entryPoint: "fragment_main",
                    targets: [{
                            format: this.canvasTextureFormat,
                        }],
                },
                primitive: {
                    topology: "triangle-list",
                },
            });
            this.clearPipeline = device.createRenderPipeline({
                layout: "auto",
                vertex: {
                    module: device.createShaderModule({
                        code: canvasRenderWGSL,
                    }),
                    entryPoint: "vertex_main",
                },
                fragment: {
                    module: device.createShaderModule({
                        code: canvasRenderWGSL,
                    }),
                    entryPoint: "fragment_clear",
                    targets: [{
                            format: this.canvasTextureFormat,
                        }],
                },
                primitive: {
                    topology: "triangle-list",
                },
            });
            this.renderSampler = device.createSampler({
                magFilter: "linear",
                minFilter: "linear",
            });
            // staging texture always be in RGBA
            this.stagingTexture = device.createTexture({
                size: [canvas.height, canvas.width, 1],
                format: "rgba8unorm",
                usage: GPUTextureUsage.TEXTURE_BINDING |
                    GPUTextureUsage.COPY_DST |
                    GPUTextureUsage.RENDER_ATTACHMENT,
            });
        }
        clear() {
            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginRenderPass({
                colorAttachments: [
                    {
                        view: this.canvasContext.getCurrentTexture().createView(),
                        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                        loadOp: "clear",
                        storeOp: "store",
                    },
                ],
            });
            passEncoder.setPipeline(this.clearPipeline);
            const renderBindingGroup = this.device.createBindGroup({
                layout: this.renderPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: this.renderSampler },
                    { binding: 1, resource: this.stagingTexture.createView() },
                ],
            });
            passEncoder.setBindGroup(0, renderBindingGroup);
            passEncoder.draw(6, 1, 0, 0);
            passEncoder.end();
            this.device.queue.submit([commandEncoder.finish()]);
        }
        draw(buffer, height, width) {
            // resize the staging texture
            if (height != this.stagingTexture.height || width != this.stagingTexture.width) {
                this.stagingTexture.destroy();
                this.stagingTexture = this.device.createTexture({
                    size: [height, width, 1],
                    format: "rgba8unorm",
                    usage: GPUTextureUsage.TEXTURE_BINDING |
                        GPUTextureUsage.COPY_DST |
                        GPUTextureUsage.RENDER_ATTACHMENT,
                });
            }
            const commandEncoder = this.device.createCommandEncoder();
            commandEncoder.copyBufferToTexture({
                buffer: buffer,
                offset: 0,
                bytesPerRow: this.stagingTexture.width * 4
            }, {
                texture: this.stagingTexture
            }, {
                width: this.stagingTexture.width,
                height: this.stagingTexture.height
            });
            const passEncoder = commandEncoder.beginRenderPass({
                colorAttachments: [
                    {
                        view: this.canvasContext.getCurrentTexture().createView(),
                        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                        loadOp: "clear",
                        storeOp: "store",
                    },
                ],
            });
            passEncoder.setPipeline(this.renderPipeline);
            const renderBindingGroup = this.device.createBindGroup({
                layout: this.renderPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: this.renderSampler },
                    { binding: 1, resource: this.stagingTexture.createView() },
                ],
            });
            passEncoder.setBindGroup(0, renderBindingGroup);
            passEncoder.draw(6, 1, 0, 0);
            passEncoder.end();
            this.device.queue.submit([commandEncoder.finish()]);
        }
        dispose() {
            this.stagingTexture.destroy();
        }
    }
    /**
     * WebGPU context
     * Manages all the webgpu resources here.
     */
    class WebGPUContext {
        constructor(memory, device) {
            // internal data
            this.bufferTable = [undefined];
            this.bufferTableFreeId = [];
            this.canvasRenderManager = undefined;
            // Pool of MAP_READ staging buffers to avoid per-copy create/destroy overhead
            this.readStagingBufferPool = [];
            this.maxReadStagingBuffers = 4;
            // Pending mapAsync promise from the last GPU→CPU copy.
            // Used in sync() as a fast path: if the last queue operation was a
            // GPU→CPU copy, awaiting its mapAsync is sufficient (no need for
            // the heavier onSubmittedWorkDone). Reset to null after any non-copy
            // queue submission so we fall back to onSubmittedWorkDone.
            this.pendingGPUToCPUCopy = null;
            // Batched command encoding: accumulate compute passes in a single encoder,
            // submit only on flush to reduce JS-native transition overhead.
            this.pendingEncoder = null;
            // Pool of uniform buffers reused across flushes. Each dispatch in a batch
            // gets its own buffer (indexed by pendingDispatchCount). The pool grows
            // as needed but buffers are never destroyed — just reused next batch.
            this.uniformBufferPool = [];
            this.uniformBufferPoolSizes = [];
            this.pendingDispatchCount = 0;
            // flags for debugging
            // stats of the runtime.
            // peak allocation
            this.peakAllocatedBytes = 0;
            // current allocation
            this.currAllocatedBytes = 0;
            // all allocation(ignoring free)
            this.allAllocatedBytes = 0;
            // shader submit counter
            this.shaderSubmitCounter = 0;
            // limite number of shaders to be submitted, useful for debugging, default to -1
            this.debugShaderSubmitLimit = -1;
            // log and sync each step
            this.debugLogFinish = false;
            this.memory = memory;
            this.device = device;
        }
        /**
         * Flush all pending compute passes by finishing and submitting the
         * accumulated command encoder.
         *
         * Must be called before:
         * - GPU→CPU readback (deviceCopyFromGPU)
         * - CPU→GPU writes (deviceCopyToGPU, copyRawBytesToBuffer)
         * - GPU↔GPU copies (deviceCopyWithinGPU)
         * - Buffer deallocation (deviceFreeDataSpace)
         * - Queue sync (sync)
         */
        flushCommands() {
            if (this.pendingEncoder) {
                this.device.queue.submit([this.pendingEncoder.finish()]);
                this.pendingEncoder = null;
                this.pendingDispatchCount = 0;
                // A compute submission is now the last queue operation, so the
                // GPU→CPU copy fast path in sync() is no longer valid.
                this.pendingGPUToCPUCopy = null;
            }
        }
        /**
         * Dispose context.
         */
        dispose() {
            var _a, _b, _c;
            this.flushCommands();
            (_a = this.canvasRenderManager) === null || _a === void 0 ? void 0 : _a.dispose();
            this.bufferTableFreeId = [];
            while (this.bufferTable.length != 0) {
                (_b = this.bufferTable.pop()) === null || _b === void 0 ? void 0 : _b.destroy();
            }
            for (const buf of this.uniformBufferPool) {
                buf.destroy();
            }
            this.uniformBufferPool.length = 0;
            this.uniformBufferPoolSizes.length = 0;
            while (this.readStagingBufferPool.length != 0) {
                (_c = this.readStagingBufferPool.pop()) === null || _c === void 0 ? void 0 : _c.buffer.destroy();
            }
            this.device.destroy();
        }
        /**
         * Wait for all pending GPU tasks to complete
         */
        sync() {
            return __awaiter(this, void 0, void 0, function* () {
                // Flush any batched compute passes before waiting on the queue.
                this.flushCommands();
                if (this.pendingGPUToCPUCopy) {
                    const p = this.pendingGPUToCPUCopy;
                    this.pendingGPUToCPUCopy = null;
                    yield p;
                }
                else {
                    yield this.device.queue.onSubmittedWorkDone();
                }
            });
        }
        /**
         * Obtain the runtime information in readable format.
         */
        runtimeStatsText() {
            let info = "peak-memory=" + Math.ceil(this.peakAllocatedBytes / (1 << 20)) + " MB";
            info += ", all-memory=" + Math.ceil(this.allAllocatedBytes / (1 << 20)) + " MB";
            info += ", shader-submissions=" + this.shaderSubmitCounter;
            return info;
        }
        /**
         * Draw image from data in storage buffer.
         * @param ptr The GPU ptr
         * @param height The height of the image.
         * @param width The width of the image.
         */
        drawImageFromBuffer(ptr, height, width) {
            if (this.canvasRenderManager == undefined) {
                throw Error("Do not have a canvas context, call bindCanvas first");
            }
            this.canvasRenderManager.draw(this.gpuBufferFromPtr(ptr), height, width);
        }
        /**
         * Copy raw bytes into buffer ptr.
         *
         * @param rawBytes The raw bytes
         * @param toPtr The target gpu buffer ptr
         * @param toOffset The beginning offset
         * @param nbytes Number of bytes
         */
        copyRawBytesToBuffer(rawBytes, toPtr, toOffset, nbytes) {
            // Flush batched compute passes before writing, to preserve execution order.
            this.flushCommands();
            this.device.queue.writeBuffer(this.gpuBufferFromPtr(toPtr), toOffset, rawBytes, 0, nbytes);
        }
        /**
         * Clear canvas
         */
        clearCanvas() {
            var _a;
            (_a = this.canvasRenderManager) === null || _a === void 0 ? void 0 : _a.clear();
        }
        /**
         * Bind a canvas element to the runtime.
         * @param canvas The HTML canvas/
         */
        bindCanvas(canvas) {
            this.canvasRenderManager = new CanvasRenderManager(this.device, canvas);
        }
        /**
         * Create a PackedFunc that runs the given shader
         * via createComputePipeline
         *
          * @param finfo The function information already parsed as a record.
         * @param code The shader data(in WGSL)
         * @returns The shader
         */
        createShader(finfo, code) {
            return this.createShadeInternal(finfo, code, false);
        }
        /**
         * Create a PackedFunc that runs the given shader asynchronously
         * via createComputePipelineAsync
         *
          * @param finfo The function information already parsed as a record.
         * @param code The shader data(in WGSL)
         * @returns The shader
         */
        createShaderAsync(finfo, code) {
            return __awaiter(this, void 0, void 0, function* () {
                return yield this.createShadeInternal(finfo, code, true);
            });
        }
        /**
         * Get a uniform buffer from the per-dispatch pool.
         *
         * Each dispatch in a batched encoder needs its own uniform buffer because
         * queue.writeBuffer() executes immediately while compute passes are deferred.
         * Reusing a shared buffer would overwrite data before earlier dispatches
         * consume it.
         *
         * The pool grows as needed. Buffers are reused across flushes (indexed by
         * dispatch position within the current batch). If the pool has no slot for
         * this dispatch, we flush first — this submits all pending passes, resets
         * pendingDispatchCount to 0, and allows reuse from the start of the pool.
         *
         * State after flush: the pending encoder and all bind group / buffer
         * references from prior dispatches are submitted and consumed. The new
         * dispatch starts a fresh encoder, so no stale state carries over.
         *
         * @param nbytes Minimum buffer size in bytes.
         * @returns A GPUBuffer with UNIFORM | COPY_DST usage, at least nbytes large.
         */
        getUniformFromPool(nbytes) {
            const dispatchIdx = this.pendingDispatchCount++;
            if (dispatchIdx < this.uniformBufferPool.length &&
                this.uniformBufferPoolSizes[dispatchIdx] >= nbytes) {
                return this.uniformBufferPool[dispatchIdx];
            }
            // Destroy old undersized buffer if it exists.
            if (dispatchIdx < this.uniformBufferPool.length) {
                this.uniformBufferPool[dispatchIdx].destroy();
            }
            const buffer = this.device.createBuffer({
                size: nbytes,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            this.uniformBufferPool[dispatchIdx] = buffer;
            this.uniformBufferPoolSizes[dispatchIdx] = nbytes;
            return buffer;
        }
        /**
         * Internal impl of createShader for both async and sync mode.
         *
          * @param finfo The function information already parsed as a record.
         * @param code The shader data(in WGSL)
         * @param asyncMode Whether use async mode.
         * @returns The shader function or promise of shader func.
         */
        createShadeInternal(finfo, code, asyncMode) {
            const dispatchToDim = [];
            let paramWriteAccess = [];
            for (let i = 0; i < finfo.launch_param_tags.length; ++i) {
                const tag = finfo.launch_param_tags[i];
                if (tag.startsWith("blockIdx.")) {
                    const target = tag.charCodeAt(tag.length - 1) - ("x".charCodeAt(0));
                    assert(target >= 0 && target < 3);
                    dispatchToDim.push(target);
                }
                else if (tag.startsWith("threadIdx.")) {
                    const target = tag.charCodeAt(tag.length - 1) - ("x".charCodeAt(0));
                    assert(target >= 0 && target < 3);
                    dispatchToDim.push(target + 3);
                }
                else if (tag.startsWith("paramWriteAccess:")) {
                    paramWriteAccess = JSON.parse(tag.substring(17));
                }
                else {
                    throw new Error("Cannot handle thread_axis " + tag);
                }
            }
            const layoutEntries = [];
            const bufferArgIndices = [];
            const podArgIndices = [];
            for (let i = 0; i < finfo.arg_types.length; ++i) {
                const dtype = finfo.arg_types[i];
                if (dtype == "handle") {
                    layoutEntries.push({
                        binding: bufferArgIndices.length,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: paramWriteAccess[bufferArgIndices.length] ? "storage" : "read-only-storage"
                        }
                    });
                    bufferArgIndices.push(i);
                }
                else if (dtype.startsWith("int") || dtype.startsWith("uint") || dtype.startsWith("float")) {
                    podArgIndices.push(i);
                }
                else {
                    throw new Error("Cannot handle argument type " + dtype + " in WebGPU shader");
                }
            }
            assert(paramWriteAccess.length == bufferArgIndices.length);
            // POD arguments are pass in the end
            layoutEntries.push({
                binding: bufferArgIndices.length,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "uniform"
                }
            });
            const bindGroupLayout = this.device.createBindGroupLayout({
                entries: layoutEntries
            });
            const pipelineLayout = this.device.createPipelineLayout({
                bindGroupLayouts: [bindGroupLayout]
            });
            // Function to create the pipeline.
            const createShaderFunc = (pipeline) => {
                const submitShader = (...args) => {
                    if (this.debugShaderSubmitLimit != -1 &&
                        this.shaderSubmitCounter >= this.debugShaderSubmitLimit) {
                        this.shaderSubmitCounter += 1;
                        return;
                    }
                    // Reuse a single command encoder across dispatches; only flush on sync/readback.
                    if (!this.pendingEncoder) {
                        this.pendingEncoder = this.device.createCommandEncoder();
                    }
                    const compute = this.pendingEncoder.beginComputePass();
                    compute.setPipeline(pipeline);
                    const bindGroupEntries = [];
                    const numBufferOrPodArgs = bufferArgIndices.length + podArgIndices.length;
                    assert(args.length == numBufferOrPodArgs + dispatchToDim.length);
                    const workDim = [1, 1, 1, 1, 1, 1];
                    for (let i = 0; i < dispatchToDim.length; ++i) {
                        workDim[dispatchToDim[i]] = args[numBufferOrPodArgs + i];
                    }
                    // get around 65535 restriction of blockIdx.x
                    if (workDim[2] != 1) {
                        throw Error("WebGPU: blockIdx.z is reserved for internal use");
                    }
                    const packDimX = workDim[0];
                    // spread thinsg out into blockIdx.z
                    if (workDim[0] >= (1 << 16)) {
                        let wl_x = workDim[0];
                        let wl_z = workDim[2];
                        while (wl_x >= (1 << 16)) {
                            if (wl_x % 2 == 0) {
                                wl_x = wl_x / 2;
                            }
                            else {
                                // pad up
                                wl_x = (wl_x + 1) / 2;
                            }
                            wl_z *= 2;
                        }
                        workDim[0] = wl_x;
                        workDim[2] = wl_z;
                        assert(wl_x * wl_z >= packDimX);
                    }
                    for (let i = 0; i < bufferArgIndices.length; ++i) {
                        bindGroupEntries.push({
                            binding: i,
                            resource: {
                                buffer: this.gpuBufferFromPtr(args[bufferArgIndices[i]])
                            }
                        });
                    }
                    const sizeOfI32 = 4;
                    const bufBytes = (podArgIndices.length + 1) * sizeOfI32;
                    const podArgBuffer = this.getUniformFromPool(bufBytes);
                    const i32View = new Int32Array(podArgIndices.length + 1);
                    const u32View = new Uint32Array(i32View.buffer);
                    const f32View = new Float32Array(i32View.buffer);
                    for (let i = 0; i < podArgIndices.length; ++i) {
                        const value = args[podArgIndices[i]];
                        const dtype = finfo.arg_types[podArgIndices[i]];
                        if (dtype.startsWith("int")) {
                            i32View[i] = value;
                        }
                        else if (dtype.startsWith("uint")) {
                            u32View[i] = value;
                        }
                        else if (dtype.startsWith("float")) {
                            f32View[i] = value;
                        }
                        else {
                            throw Error("Unknown pod dtype " + dtype);
                        }
                    }
                    // always pass in dim z launching grid size in
                    u32View[podArgIndices.length] = packDimX;
                    this.device.queue.writeBuffer(podArgBuffer, 0, i32View.buffer);
                    bindGroupEntries.push({
                        binding: bufferArgIndices.length,
                        resource: {
                            buffer: podArgBuffer,
                            size: i32View.buffer.byteLength
                        }
                    });
                    compute.setBindGroup(0, this.device.createBindGroup({
                        layout: bindGroupLayout,
                        entries: bindGroupEntries
                    }));
                    compute.dispatchWorkgroups(workDim[0], workDim[1], workDim[2]);
                    compute.end();
                    // In debug mode, flush immediately so we can observe each submission.
                    if (this.debugLogFinish) {
                        this.flushCommands();
                        const currCounter = this.shaderSubmitCounter;
                        this.device.queue.onSubmittedWorkDone().then(() => {
                            console.log("[" + currCounter + "][Debug] finish shader" + finfo.name);
                        });
                    }
                    this.shaderSubmitCounter += 1;
                };
                return submitShader;
            };
            const shaderModule = this.device.createShaderModule({
                code: code,
                compilationHints: [
                    {
                        entryPoint: "main",
                        layout: pipelineLayout
                    }
                ]
            });
            if (asyncMode) {
                return this.device.createComputePipelineAsync({
                    layout: pipelineLayout,
                    compute: {
                        module: shaderModule,
                        entryPoint: finfo.name
                    }
                }).then((pipeline) => {
                    return createShaderFunc(pipeline);
                });
            }
            else {
                const pipeline = this.device.createComputePipeline({
                    layout: pipelineLayout,
                    compute: {
                        module: shaderModule,
                        entryPoint: finfo.name
                    }
                });
                return createShaderFunc(pipeline);
            }
        }
        /**
         * Get the device API according to its name
          * @param name The name of the API.
         * @returns The corresponding device api.
         */
        getDeviceAPI(name) {
            if (name == "deviceAllocDataSpace") {
                return (nbytes) => {
                    return this.deviceAllocDataSpace(nbytes);
                };
            }
            else if (name == "deviceFreeDataSpace") {
                return (ptr) => {
                    return this.deviceFreeDataSpace(ptr);
                };
            }
            else if (name == "deviceCopyToGPU") {
                return (from, to, toOffset, nbytes) => {
                    this.deviceCopyToGPU(from, to, toOffset, nbytes);
                };
            }
            else if (name == "deviceCopyFromGPU") {
                return (from, fromOffset, to, nbytes) => {
                    this.deviceCopyFromGPU(from, fromOffset, to, nbytes);
                };
            }
            else if (name == "deviceCopyWithinGPU") {
                return (from, fromOffset, to, toOffset, nbytes) => {
                    this.deviceCopyWithinGPU(from, fromOffset, to, toOffset, nbytes);
                };
            }
            else {
                throw new Error("Unknown DeviceAPI function " + name);
            }
        }
        // DeviceAPI
        deviceAllocDataSpace(nbytes) {
            // allocate 0 bytes buffer as 1 bytes buffer.
            if (nbytes == 0) {
                nbytes = 1;
            }
            const buffer = tryCreateBuffer(this.device, {
                size: nbytes,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            });
            this.currAllocatedBytes += nbytes;
            this.allAllocatedBytes += nbytes;
            if (this.currAllocatedBytes > this.peakAllocatedBytes) {
                this.peakAllocatedBytes = this.currAllocatedBytes;
            }
            const ptr = this.attachToBufferTable(buffer);
            return ptr;
        }
        deviceFreeDataSpace(ptr) {
            const idx = ptr;
            const buffer = this.bufferTable[idx];
            this.bufferTable[idx] = undefined;
            assert(buffer !== undefined);
            this.bufferTableFreeId.push(idx);
            this.currAllocatedBytes -= buffer.size;
            // Flush any pending compute passes that may reference this buffer
            // before destroying it, otherwise queue.submit() will fail with
            // "buffer used in submit while destroyed".
            this.flushCommands();
            buffer.destroy();
        }
        deviceCopyToGPU(from, to, toOffset, nbytes) {
            // Flush batched compute passes before writing to a GPU buffer,
            // otherwise the write may be reordered before pending dispatches
            // that read from the same buffer.
            this.flushCommands();
            let rawBytes = this.memory.loadRawBytes(from, nbytes);
            if (rawBytes.length % 4 !== 0) {
                // writeBuffer requires length to be multiples of 4, so we pad here
                const toPad = 4 - rawBytes.length % 4;
                const padded = new Uint8Array(rawBytes.length + toPad);
                padded.set(rawBytes);
                rawBytes = padded;
                nbytes = nbytes + toPad;
            }
            this.device.queue.writeBuffer(this.gpuBufferFromPtr(to), toOffset, rawBytes, 0, nbytes);
        }
        /**
         * Get a MAP_READ staging buffer from the pool, or create one if none fits.
         * Uses first-fit-by-size: returns the first pooled buffer >= nbytes.
         */
        getOrCreateReadStagingBuffer(nbytes) {
            for (let i = 0; i < this.readStagingBufferPool.length; i++) {
                if (this.readStagingBufferPool[i].size >= nbytes) {
                    const entry = this.readStagingBufferPool.splice(i, 1)[0];
                    return entry.buffer;
                }
            }
            return tryCreateBuffer(this.device, {
                size: nbytes,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });
        }
        /**
         * Return a MAP_READ staging buffer to the pool for reuse.
         * Evicts the smallest buffer if the pool is full.
         */
        recycleReadStagingBuffer(buf) {
            buf.unmap();
            if (this.readStagingBufferPool.length >= this.maxReadStagingBuffers) {
                // Evict smallest buffer to make room
                let minIdx = 0;
                for (let i = 1; i < this.readStagingBufferPool.length; i++) {
                    if (this.readStagingBufferPool[i].size < this.readStagingBufferPool[minIdx].size) {
                        minIdx = i;
                    }
                }
                this.readStagingBufferPool.splice(minIdx, 1)[0].buffer.destroy();
            }
            this.readStagingBufferPool.push({ buffer: buf, size: buf.size });
        }
        deviceCopyFromGPU(from, fromOffset, to, nbytes) {
            // Flush batched compute passes before the readback copy.
            this.flushCommands();
            // WebGPU copy/map sizes must be 4-byte multiples, but small tensors
            // (e.g. a 3-element f16 WDL head = 6 bytes) are not. Pad the copy within
            // the source buffer's real size and store back only the requested bytes.
            const fromBuffer = this.gpuBufferFromPtr(from);
            const copyNbytes = Math.min((nbytes + 3) & -4, fromBuffer.size - fromOffset);
            if (copyNbytes % 4 !== 0 || copyNbytes < nbytes) {
                throw new Error(`GPU readback needs a 4-byte aligned copy; nbytes=${nbytes} cannot be padded ` +
                    `within the source buffer (size=${fromBuffer.size}, offset=${fromOffset})`);
            }
            const gpuTemp = this.getOrCreateReadStagingBuffer(copyNbytes);
            const copyEncoder = this.device.createCommandEncoder();
            copyEncoder.copyBufferToBuffer(fromBuffer, fromOffset, gpuTemp, 0, copyNbytes);
            const copyCommands = copyEncoder.finish();
            this.device.queue.submit([copyCommands]);
            const readPromise = gpuTemp.mapAsync(GPUMapMode.READ).then(() => {
                const data = gpuTemp.getMappedRange(0, copyNbytes);
                this.memory.storeRawBytes(to, new Uint8Array(data, 0, nbytes));
                this.recycleReadStagingBuffer(gpuTemp);
            });
            // Chain with any existing pending read so sync() awaits all of them.
            this.pendingGPUToCPUCopy = this.pendingGPUToCPUCopy
                ? this.pendingGPUToCPUCopy.then(() => readPromise)
                : readPromise;
        }
        deviceCopyWithinGPU(from, fromOffset, to, toOffset, nbytes) {
            // Flush batched compute passes before the GPU-to-GPU copy.
            this.flushCommands();
            const copyEncoder = this.device.createCommandEncoder();
            copyEncoder.copyBufferToBuffer(this.gpuBufferFromPtr(from), fromOffset, this.gpuBufferFromPtr(to), toOffset, nbytes);
            const copyCommands = copyEncoder.finish();
            this.device.queue.submit([copyCommands]);
        }
        gpuBufferFromPtr(ptr) {
            const buffer = this.bufferTable[ptr];
            assert(buffer !== undefined);
            return buffer;
        }
        attachToBufferTable(buffer) {
            if (this.bufferTableFreeId.length != 0) {
                const idx = this.bufferTableFreeId.pop();
                this.bufferTable[idx] = buffer;
                return idx;
            }
            else {
                const idx = this.bufferTable.length;
                this.bufferTable.push(buffer);
                return idx;
            }
        }
    }

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
    /**
     * A generic LRU (Least Recently Used) cache with bounded size.
     *
     * Entries are evicted in insertion order when the cache exceeds `maxSize`.
     * Uses a Map to maintain insertion order for O(1) LRU eviction.
     *
     * @typeParam K - Cache key type.
     * @typeParam V - Cache value type.
     */
    class LRUCache {
        constructor(maxSize, onEvict) {
            this.cache = new Map();
            this.maxSize = maxSize;
            this.onEvict = onEvict;
        }
        /**
         * Get a value from the cache, constructing it via `constructor` on miss.
         *
         * On hit: moves the entry to most-recently-used position and returns it.
         * On miss: calls `constructor()` to create the value, inserts it, and
         * returns it. If the cache is full, the least-recently-used entry is
         * evicted first.
         *
         * @param key The cache key.
         * @param constructor Factory function called on cache miss to produce the value.
         * @returns The cached or newly constructed value.
         */
        get(key, constructor) {
            const existing = this.cache.get(key);
            if (existing !== undefined) {
                // Move to most-recently-used position
                this.cache.delete(key);
                this.cache.set(key, existing);
                return existing;
            }
            // Evict LRU entry if at capacity
            if (this.cache.size >= this.maxSize) {
                const oldest = this.cache.keys().next().value;
                if (oldest !== undefined) {
                    if (this.onEvict) {
                        this.onEvict(oldest, this.cache.get(oldest));
                    }
                    this.cache.delete(oldest);
                }
            }
            const value = constructor();
            this.cache.set(key, value);
            return value;
        }
        /**
         * Check whether eviction would be needed for a new entry.
         *
         * Useful when the caller needs to perform side effects before eviction
         * (e.g. flushing pending GPU commands before destroying an evicted buffer).
         *
         * @param key The key to check.
         * @returns true if inserting `key` would trigger eviction of another entry.
         */
        needEviction(key) {
            if (this.cache.has(key))
                return false;
            return this.cache.size >= this.maxSize;
        }
        /**
         * Clear all cached entries.
         *
         * Does not dispose values — the caller is responsible for cleanup
         * (e.g. destroying GPU buffers) before calling invalidate.
         */
        invalidate() {
            this.cache.clear();
        }
        /** Number of entries currently in the cache. */
        get size() {
            return this.cache.size;
        }
        /** Iterate over all cached values (for disposal). */
        values() {
            return this.cache.values();
        }
    }
    /**
     * CacheState manages domain-specific caches for the WebGPU runtime.
     *
     * Currently contains:
     * - **shapeCache**: Caches TVM ShapeTuple objects keyed by dimension string.
     *   - Why: `makeShapeTuple()` is called on every tensor operation, crossing
     *     the JS→WASM FFI boundary each time. During LLM decode, the same shapes
     *     repeat every token (e.g. [1,32,128]), so caching avoids thousands of
     *     redundant FFI round-trips.
     *   - Invalidation: Never. Shape tuples are immutable value objects that
     *     remain valid for the lifetime of the TVM instance.
     *
     * Future additions (follow-up PR):
     * - **uniformCache**: Caches GPU uniform buffers keyed by content hash.
     *   - Why: Many dispatches use identical scalar arguments (matrix dims, etc.).
     *     Reusing the buffer avoids `createBuffer` + `writeBuffer` overhead.
     *   - Invalidation: Must invalidate on any GPU buffer deallocation, since
     *     buffer pointers can be reused by the allocator, making cached entries
     *     that reference the old buffer stale.
     */
    class CacheState {
        constructor(shapeCacheSize = 256) {
            this.shapeCache = new LRUCache(shapeCacheSize, (_key, value) => value.dispose());
        }
        /**
         * Compute the cache key for a shape tuple.
         *
         * @param shape Array of dimension values.
         * @returns String key suitable for shapeCache lookup.
         */
        static computeShapeKey(shape) {
            return shape.toString();
        }
        /**
         * Dispose all cached objects and clear all caches.
         */
        dispose() {
            for (const obj of this.shapeCache.values()) {
                obj.dispose();
            }
            this.shapeCache.invalidate();
        }
    }

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
    const HASH_ALGORITHM$1 = "SHA-256";
    const OPFS_STORE_ROOT_DIRECTORY = "tvmjs-opfs-store";
    class OPFSStore {
        constructor(scope, accessMode = "async") {
            this.scope = scope;
            this.requestedAccessMode = accessMode;
            this.accessMode = OPFSStore.resolveAccessMode(accessMode);
        }
        static isAvailable() {
            const storage = OPFSStore.getStorageManager();
            return storage !== undefined && typeof storage.getDirectory === "function";
        }
        static resolveAccessMode(accessMode) {
            if (accessMode !== "auto") {
                return accessMode;
            }
            return OPFSStore.isDedicatedWorkerWithSyncAccessHandle()
                ? "sync"
                : "async";
        }
        has(url) {
            return __awaiter(this, void 0, void 0, function* () {
                try {
                    const entry = yield this.getStoredEntry(url);
                    if (entry === undefined) {
                        return false;
                    }
                    return this.hasExpectedPayloadSize(entry);
                }
                catch (err) {
                    if (this.handleCacheMissStateError(err)) {
                        return false;
                    }
                    throw err;
                }
            });
        }
        read(url) {
            return __awaiter(this, void 0, void 0, function* () {
                try {
                    const entry = yield this.getStoredEntry(url);
                    if (entry === undefined) {
                        return undefined;
                    }
                    const blob = yield entry.payloadHandle.getFile();
                    if (blob.size !== entry.record.nbytes) {
                        return undefined;
                    }
                    return new Response(blob, this.getResponseInit(entry.record));
                }
                catch (err) {
                    if (this.handleCacheMissStateError(err)) {
                        return undefined;
                    }
                    throw err;
                }
            });
        }
        readArrayBuffer(url) {
            return __awaiter(this, void 0, void 0, function* () {
                try {
                    const entry = yield this.getStoredEntry(url);
                    if (entry === undefined) {
                        return undefined;
                    }
                    const payload = yield this.readPayload(entry.payloadHandle);
                    return payload.byteLength === entry.record.nbytes ? payload : undefined;
                }
                catch (err) {
                    if (this.handleCacheMissStateError(err)) {
                        return undefined;
                    }
                    throw err;
                }
            });
        }
        write(url, response) {
            return __awaiter(this, void 0, void 0, function* () {
                var _a;
                try {
                    const directory = yield this.getScopedDirectory();
                    const baseName = yield this.hashUrl(url);
                    yield this.removeEntryIfExists(directory, this.getRecordFilename(baseName));
                    const payloadHandle = yield directory.getFileHandle(this.getPayloadFilename(baseName), { create: true });
                    const nbytes = yield this.writePayload(payloadHandle, response);
                    const recordHandle = yield directory.getFileHandle(this.getRecordFilename(baseName), { create: true });
                    const record = {
                        url,
                        nbytes,
                        contentType: (_a = response.headers.get("content-type")) !== null && _a !== void 0 ? _a : undefined,
                    };
                    yield this.writeRecord(recordHandle, record);
                }
                catch (err) {
                    this.resetDirectoryOnInvalidStateError(err);
                    throw err;
                }
            });
        }
        remove(url) {
            return __awaiter(this, void 0, void 0, function* () {
                try {
                    const directory = yield this.getScopedDirectory();
                    const baseName = yield this.hashUrl(url);
                    yield this.removeEntryIfExists(directory, this.getPayloadFilename(baseName));
                    yield this.removeEntryIfExists(directory, this.getRecordFilename(baseName));
                }
                catch (err) {
                    this.resetDirectoryOnInvalidStateError(err);
                    throw err;
                }
            });
        }
        getStoredEntry(url) {
            return __awaiter(this, void 0, void 0, function* () {
                const directory = yield this.getScopedDirectory();
                const baseName = yield this.hashUrl(url);
                const recordHandle = yield this.getFileHandleIfExists(directory, this.getRecordFilename(baseName), false);
                if (recordHandle === undefined) {
                    return undefined;
                }
                const record = yield this.readRecord(recordHandle);
                if (record === undefined || record.url !== url) {
                    return undefined;
                }
                const payloadHandle = yield this.getFileHandleIfExists(directory, this.getPayloadFilename(baseName), false);
                return payloadHandle === undefined ? undefined : { payloadHandle, record };
            });
        }
        static getStorageManager() {
            if (typeof navigator === "undefined") {
                return undefined;
            }
            return navigator.storage;
        }
        static isDedicatedWorkerWithSyncAccessHandle() {
            var _a, _b;
            const scope = globalThis;
            return (typeof scope.DedicatedWorkerGlobalScope === "function" &&
                globalThis instanceof scope.DedicatedWorkerGlobalScope &&
                typeof ((_b = (_a = scope.FileSystemFileHandle) === null || _a === void 0 ? void 0 : _a.prototype) === null || _b === void 0 ? void 0 : _b.createSyncAccessHandle) ===
                    "function");
        }
        getScopedDirectory() {
            return __awaiter(this, void 0, void 0, function* () {
                if (this.directoryPromise !== undefined) {
                    return this.directoryPromise;
                }
                // Cache scoped directory handle to avoid repeated tree traversal
                this.directoryPromise = (() => __awaiter(this, void 0, void 0, function* () {
                    const storage = OPFSStore.getStorageManager();
                    if (storage === undefined || typeof storage.getDirectory !== "function") {
                        throw new Error("OPFSStore: OPFS API unavailable.");
                    }
                    let directory = yield storage.getDirectory();
                    directory = yield directory.getDirectoryHandle(OPFS_STORE_ROOT_DIRECTORY, {
                        create: true,
                    });
                    const scopeParts = this.scope.split("/").filter((part) => part.length > 0);
                    for (const part of scopeParts) {
                        directory = yield directory.getDirectoryHandle(encodeURIComponent(part), { create: true });
                    }
                    return directory;
                }))();
                return this.directoryPromise;
            });
        }
        readRecord(fileHandle) {
            return __awaiter(this, void 0, void 0, function* () {
                try {
                    const text = yield (yield fileHandle.getFile()).text();
                    const parsed = JSON.parse(text);
                    if (parsed === undefined ||
                        parsed === null ||
                        typeof parsed !== "object" ||
                        typeof parsed.url !== "string" ||
                        !Number.isSafeInteger(parsed.nbytes) ||
                        parsed.nbytes < 0) {
                        return undefined;
                    }
                    const record = {
                        url: parsed.url,
                        nbytes: parsed.nbytes,
                    };
                    if (typeof parsed.contentType === "string") {
                        record.contentType = parsed.contentType;
                    }
                    return record;
                }
                catch (err) {
                    if (OPFSStore.getErrorName(err) === "SyntaxError" ||
                        this.handleCacheMissStateError(err)) {
                        return undefined;
                    }
                    throw err;
                }
            });
        }
        getResponseInit(record) {
            return record.contentType !== undefined
                ? { headers: { "content-type": record.contentType } }
                : undefined;
        }
        writeRecord(handle, record) {
            return __awaiter(this, void 0, void 0, function* () {
                const writable = yield handle.createWritable();
                try {
                    yield writable.write(new TextEncoder().encode(JSON.stringify(record)));
                    yield writable.close();
                }
                catch (err) {
                    try {
                        yield writable.abort();
                    }
                    catch (_a) {
                        // Preserve the original write error.
                    }
                    throw err;
                }
            });
        }
        readPayload(handle) {
            return __awaiter(this, void 0, void 0, function* () {
                const syncHandle = yield this.openSyncAccessHandle(handle, "read-only");
                return syncHandle !== undefined
                    ? this.readPayloadWithSyncHandle(syncHandle)
                    : (yield handle.getFile()).arrayBuffer();
            });
        }
        hasExpectedPayloadSize(entry) {
            return __awaiter(this, void 0, void 0, function* () {
                if (this.accessMode === "sync") {
                    const syncHandle = yield this.openSyncAccessHandle(entry.payloadHandle, "read-only");
                    if (syncHandle !== undefined) {
                        try {
                            return syncHandle.getSize() === entry.record.nbytes;
                        }
                        finally {
                            syncHandle.close();
                        }
                    }
                }
                const blob = yield entry.payloadHandle.getFile();
                return blob.size === entry.record.nbytes;
            });
        }
        writePayload(handle, response) {
            return __awaiter(this, void 0, void 0, function* () {
                const syncHandle = yield this.openSyncAccessHandle(handle, "readwrite");
                if (syncHandle !== undefined) {
                    return this.writePayloadWithSyncHandle(syncHandle, response);
                }
                return this.writePayloadWithWritable(handle, response);
            });
        }
        writePayloadWithWritable(handle, response) {
            return __awaiter(this, void 0, void 0, function* () {
                const writable = yield handle.createWritable();
                try {
                    if (response.body !== null) {
                        let nbytes = 0;
                        const reader = response.body.getReader();
                        try {
                            while (true) {
                                const { done, value } = yield reader.read();
                                if (done) {
                                    break;
                                }
                                yield writable.write(value);
                                nbytes += value.byteLength;
                            }
                        }
                        finally {
                            reader.releaseLock();
                        }
                        yield writable.close();
                        return nbytes;
                    }
                    const payload = yield response.arrayBuffer();
                    yield writable.write(payload);
                    yield writable.close();
                    return payload.byteLength;
                }
                catch (err) {
                    try {
                        yield writable.abort();
                    }
                    catch (_a) {
                        // Preserve the original write error.
                    }
                    throw err;
                }
            });
        }
        readPayloadWithSyncHandle(syncHandle) {
            try {
                const size = syncHandle.getSize();
                const payload = new ArrayBuffer(size);
                syncHandle.read(new Uint8Array(payload), { at: 0 });
                return payload;
            }
            finally {
                syncHandle.close();
            }
        }
        writePayloadWithSyncHandle(syncHandle, response) {
            return __awaiter(this, void 0, void 0, function* () {
                try {
                    syncHandle.truncate(0);
                    let offset = 0;
                    if (response.body !== null) {
                        const reader = response.body.getReader();
                        try {
                            while (true) {
                                const { done, value } = yield reader.read();
                                if (done) {
                                    break;
                                }
                                syncHandle.write(value, { at: offset });
                                offset += value.byteLength;
                            }
                        }
                        finally {
                            reader.releaseLock();
                        }
                    }
                    else {
                        const payload = yield response.arrayBuffer();
                        syncHandle.write(new Uint8Array(payload), { at: 0 });
                        offset = payload.byteLength;
                    }
                    syncHandle.flush();
                    return offset;
                }
                finally {
                    syncHandle.close();
                }
            });
        }
        openSyncAccessHandle(handle, mode) {
            return __awaiter(this, void 0, void 0, function* () {
                if (this.accessMode === "async") {
                    return undefined;
                }
                if (typeof handle.createSyncAccessHandle !== "function") {
                    throw this.createSyncUnavailableError();
                }
                try {
                    return yield handle.createSyncAccessHandle({ mode });
                }
                catch (err) {
                    const isLockContention = OPFSStore.getErrorName(err) === "NoModificationAllowedError";
                    if (this.requestedAccessMode === "auto" && isLockContention) {
                        return undefined;
                    }
                    throw err;
                }
            });
        }
        getFileHandleIfExists(directory, filename, create) {
            return __awaiter(this, void 0, void 0, function* () {
                try {
                    return yield directory.getFileHandle(filename, { create });
                }
                catch (err) {
                    if (OPFSStore.isNotFoundError(err)) {
                        // NotFound maps to cache miss semantics
                        return undefined;
                    }
                    throw err;
                }
            });
        }
        removeEntryIfExists(directory, filename) {
            return __awaiter(this, void 0, void 0, function* () {
                try {
                    yield directory.removeEntry(filename);
                }
                catch (err) {
                    if (OPFSStore.isNotFoundError(err)) {
                        // Delete is intentionally idempotent for missing entries
                        return;
                    }
                    throw err;
                }
            });
        }
        hashUrl(url) {
            return __awaiter(this, void 0, void 0, function* () {
                const textEncoder = new TextEncoder();
                const input = textEncoder.encode(url);
                if (typeof crypto === "undefined" ||
                    crypto.subtle === undefined ||
                    typeof crypto.subtle.digest !== "function") {
                    throw new Error("OPFSStore: crypto.subtle.digest is unavailable.");
                }
                const digest = yield crypto.subtle.digest(HASH_ALGORITHM$1, input);
                return Array.from(new Uint8Array(digest))
                    .map((byte) => byte.toString(16).padStart(2, "0"))
                    .join("");
            });
        }
        static isNotFoundError(err) {
            return OPFSStore.getErrorName(err) === "NotFoundError";
        }
        static isCacheMissStateError(err) {
            const name = OPFSStore.getErrorName(err);
            return name === "NotFoundError" || name === "InvalidStateError";
        }
        handleCacheMissStateError(err) {
            if (!OPFSStore.isCacheMissStateError(err)) {
                return false;
            }
            this.resetDirectoryOnInvalidStateError(err);
            return true;
        }
        resetDirectoryOnInvalidStateError(err) {
            if (OPFSStore.getErrorName(err) === "InvalidStateError") {
                this.directoryPromise = undefined;
            }
        }
        static getErrorName(err) {
            if (err && typeof err === "object" && "name" in err) {
                const name = err.name;
                return typeof name === "string" ? name : undefined;
            }
            return undefined;
        }
        getPayloadFilename(baseName) {
            return `${baseName}.bin`;
        }
        getRecordFilename(baseName) {
            return `${baseName}.record.json`;
        }
        createSyncUnavailableError() {
            const err = new Error("OPFSStore: createSyncAccessHandle unavailable; sync OPFS access requires a supported dedicated worker context.");
            err.name = "NotSupportedError";
            return err;
        }
    }

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
    const HASH_ALGORITHM = "SHA-256";
    const DEFAULT_FETCH_OPTIONS = { method: "GET" };
    const COS_HASH_META_CACHE = "tvmjs-cos-hash-meta";
    let crossOriginFallbackWarningLogged = false;
    const GLOBAL_HASH_CACHE = new Map();
    class CrossOriginStorage {
        constructor() {
            this.hashCache = GLOBAL_HASH_CACHE;
        }
        static isAvailable() {
            if (typeof navigator === "undefined") {
                return false;
            }
            return navigator.crossOriginStorage !== undefined;
        }
        match(request) {
            return __awaiter(this, void 0, void 0, function* () {
                const url = this.normalizeRequest(request);
                const hash = yield this.resolveHashDescriptor(url);
                if (!hash) {
                    return undefined;
                }
                try {
                    const api = this.getApi();
                    if (!api) {
                        return undefined;
                    }
                    const handles = yield api.requestFileHandles([hash]);
                    const handle = handles[0];
                    if (!handle) {
                        return undefined;
                    }
                    const blob = yield handle.getFile();
                    return new Response(blob);
                }
                catch (_a) {
                    return undefined;
                }
            });
        }
        put(request, response) {
            return __awaiter(this, void 0, void 0, function* () {
                const url = this.normalizeRequest(request);
                const blob = yield response.blob();
                const hash = yield this.getBlobHash(blob);
                const api = this.getApi();
                if (!api) {
                    throw new Error("Cross-origin storage API unavailable.");
                }
                const handles = yield api.requestFileHandles([hash], { create: true });
                const handle = handles[0];
                if (!handle) {
                    throw new Error("Cross-origin storage API returned no handles.");
                }
                const writableStream = yield handle.createWritable();
                yield writableStream.write(blob);
                yield writableStream.close();
                this.hashCache.set(url, hash);
                yield this.persistHashEntry(url, hash);
            });
        }
        delete(_request) {
            return __awaiter(this, void 0, void 0, function* () {
                // Cross-origin storage extension currently has no delete API.
                return;
            });
        }
        getApi() {
            if (!CrossOriginStorage.isAvailable()) {
                return undefined;
            }
            return navigator.crossOriginStorage;
        }
        normalizeRequest(request) {
            if (typeof request === "string") {
                return request;
            }
            if (request instanceof URL) {
                return request.href;
            }
            if (request instanceof Request) {
                return request.url;
            }
            if (request && typeof request.url === "string") {
                return request.url;
            }
            throw new Error("CrossOriginStorage: Unsupported request type.");
        }
        persistHashEntry(url, hash) {
            return __awaiter(this, void 0, void 0, function* () {
                try {
                    if (typeof caches === "undefined") {
                        return;
                    }
                    const store = yield caches.open(COS_HASH_META_CACHE);
                    yield store.put(url, new Response(JSON.stringify(hash)));
                }
                catch (_a) {
                    // best-effort: ignore storage errors
                }
            });
        }
        loadPersistedHashEntry(url) {
            return __awaiter(this, void 0, void 0, function* () {
                try {
                    if (typeof caches === "undefined") {
                        return null;
                    }
                    const store = yield caches.open(COS_HASH_META_CACHE);
                    const response = yield store.match(url);
                    if (!response) {
                        return null;
                    }
                    return JSON.parse(yield response.text());
                }
                catch (_a) {
                    return null;
                }
            });
        }
        resolveHashDescriptor(url) {
            return __awaiter(this, void 0, void 0, function* () {
                const cached = this.hashCache.get(url);
                if (cached) {
                    return cached;
                }
                // Check persistent store before falling back to network-based hash extraction.
                // This covers non-LFS files (JSON configs, tokenizers) and non-HuggingFace URLs
                // (e.g. GitHub raw .wasm files) whose hashes were computed from blob content on a
                // previous visit and persisted to the Cache API.
                const persisted = yield this.loadPersistedHashEntry(url);
                if (persisted) {
                    this.hashCache.set(url, persisted);
                    return persisted;
                }
                const hashValue = yield this.getFileHash(url);
                if (!hashValue) {
                    return null;
                }
                const descriptor = {
                    algorithm: HASH_ALGORITHM,
                    value: hashValue,
                };
                this.hashCache.set(url, descriptor);
                // Persist pointer-derived hashes so subsequent visits skip the LFS pointer
                // network request (especially important for models with many shards).
                yield this.persistHashEntry(url, descriptor);
                return descriptor;
            });
        }
        getFileHash(url) {
            return __awaiter(this, void 0, void 0, function* () {
                if (/\/resolve\//.test(url)) {
                    const pointerHash = yield this.extractHashFromPointer(url);
                    if (pointerHash) {
                        return pointerHash;
                    }
                }
                return null;
            });
        }
        extractHashFromPointer(url) {
            return __awaiter(this, void 0, void 0, function* () {
                const rawUrl = url.replace(/\/resolve\//, "/raw/");
                try {
                    const text = yield fetch(rawUrl).then((res) => res.text());
                    if (!text.includes("oid sha256:")) {
                        return null;
                    }
                    const match = text.match(/oid sha256:([A-Fa-f0-9]+)/);
                    return match ? match[1] : null;
                }
                catch (_a) {
                    return null;
                }
            });
        }
        getBlobHash(blob) {
            return __awaiter(this, void 0, void 0, function* () {
                const arrayBuffer = yield blob.arrayBuffer();
                const hashBuffer = yield crypto.subtle.digest(HASH_ALGORITHM, arrayBuffer);
                const hashArray = Array.from(new Uint8Array(hashBuffer));
                const hashHex = hashArray
                    .map((byte) => byte.toString(16).padStart(2, "0"))
                    .join("");
                return {
                    algorithm: HASH_ALGORITHM,
                    value: hashHex,
                };
            });
        }
    }
    /**
     * Cache to store model related data, implemented with the Cache API.
     */
    class ArtifactCache {
        constructor(scope) {
            this.scope = scope;
        }
        /**
         * Convert the Response object to the expected storetype instead
         */
        responseTostoretype(response, storetype) {
            return __awaiter(this, void 0, void 0, function* () {
                if (storetype === undefined) {
                    return response;
                }
                else if (storetype.toLowerCase() === "json") {
                    return yield response.json();
                }
                else if (storetype.toLowerCase() === "arraybuffer") {
                    return yield response.arrayBuffer();
                }
                else {
                    console.error("Unknown storage type " + storetype + ", returning raw response");
                    return response;
                }
            });
        }
        /**
         * fetch the corresponding url object in response or stored object format
         * @param url url
         * @param storetype the storage type for indexedDB
         * @param signal an optional abort signal to abort fetching
         * @returns response in json, arraybuffer or pure response format
         */
        fetchWithCache(url, storetype, signal) {
            return __awaiter(this, void 0, void 0, function* () {
                yield this.addToCache(url, storetype, signal);
                const result = yield this.cache.match(new Request(url));
                if (result === undefined) {
                    // Already called `addToCache()`, should expect the request in cache.
                    throw Error("Cannot fetch " + url);
                }
                return yield this.responseTostoretype(result, storetype);
            });
        }
        addToCache(url, storetype, signal) {
            return __awaiter(this, void 0, void 0, function* () {
                const request = new Request(url, signal ? { signal } : undefined);
                if (this.cache === undefined) {
                    this.cache = yield caches.open(this.scope);
                }
                const result = yield this.cache.match(request);
                if (result === undefined) {
                    yield this.cache.add(request);
                }
            });
        }
        /**
         * Determine if all keys exist in the cache
         * @param keys the url key list of the strings
         * @returns boolean value indicate if all keys are in cache
         */
        hasAllKeys(keys) {
            return __awaiter(this, void 0, void 0, function* () {
                if (this.cache === undefined) {
                    this.cache = yield caches.open(this.scope);
                }
                return this.cache.keys()
                    .then(requests => requests.map(request => request.url))
                    .then(cacheKeys => keys.every(key => cacheKeys.indexOf(key) !== -1))
                    .catch(() => false);
            });
        }
        /**
         * Delete the corresponding url object in cache
         * @param url the corresponding url object to be deleted
         */
        deleteInCache(url) {
            return __awaiter(this, void 0, void 0, function* () {
                if (this.cache === undefined) {
                    this.cache = yield caches.open(this.scope);
                }
                yield this.cache.delete(url);
            });
        }
    }
    /**
     * Cache by IndexedDB to support caching model data
     */
    class ArtifactIndexedDBCache {
        constructor(dbName) {
            this.dbVersion = 1;
            this.dbName = dbName;
        }
        /**
         * Init the indexed DB database if it is not initialized.
         */
        initDB() {
            return __awaiter(this, void 0, void 0, function* () {
                if (this.db != null) {
                    return; // the db is already inialized
                }
                return new Promise((resolve, reject) => {
                    const request = indexedDB.open(this.dbName, this.dbVersion);
                    request.onupgradeneeded = (event) => {
                        this.db = event.target.result;
                        if (!this.db.objectStoreNames.contains('urls')) {
                            this.db.createObjectStore('urls', { keyPath: 'url' });
                        }
                    };
                    request.onsuccess = (event) => {
                        this.db = event.target.result;
                        resolve();
                    };
                    request.onerror = (event) => {
                        console.error("Database error: ", event.target.error);
                        reject(event.target.error);
                    };
                });
            });
        }
        /**
         * Check if current url object is in indexedDB or not
         * @param url the url link
         * @returns boolean indicate if url object in indexedDB
         */
        isUrlInDB(url) {
            return __awaiter(this, void 0, void 0, function* () {
                return new Promise((resolve, reject) => {
                    var _a;
                    const transaction = (_a = this.db) === null || _a === void 0 ? void 0 : _a.transaction(['urls'], 'readonly');
                    if (transaction === undefined) {
                        return false;
                    }
                    const store = transaction.objectStore('urls');
                    const request = store.get(url);
                    request.onsuccess = () => {
                        resolve(request.result !== undefined);
                    };
                    request.onerror = (event) => {
                        reject(event.target.error);
                    };
                });
            });
        }
        asyncGetHelper(url) {
            return __awaiter(this, void 0, void 0, function* () {
                return new Promise((resolve, reject) => {
                    var _a;
                    let result;
                    const transaction = (_a = this.db) === null || _a === void 0 ? void 0 : _a.transaction(['urls'], 'readonly');
                    if (transaction === undefined) {
                        return false;
                    }
                    transaction.oncomplete = () => resolve(result);
                    transaction.onerror = () => reject(transaction.error);
                    const objectStore = transaction.objectStore('urls');
                    const getRequest = objectStore.get(url);
                    getRequest.onsuccess = () => {
                        result = getRequest.result;
                    };
                });
            });
        }
        fetchWithCache(url, storetype, signal) {
            return __awaiter(this, void 0, void 0, function* () {
                yield this.addToCache(url, storetype, signal);
                let result = yield this.asyncGetHelper(url);
                if (result === null) {
                    // previously null data in cache or somehow failed to add to cache, delete and retry
                    yield this.deleteInCache(url);
                    yield this.addToCache(url, storetype);
                    result = yield this.asyncGetHelper(url);
                }
                if (result != null && typeof result === "object" && "data" in result) {
                    // `storetype` not used here because the data stored in indexedDB is already in that type
                    return result.data;
                }
                throw Error("ArtifactIndexedDBCache failed to fetch: " + url);
            });
        }
        addToIndexedDB(url, response, storetype) {
            return __awaiter(this, void 0, void 0, function* () {
                yield this.initDB();
                let data;
                // IndexedDB, unlike the Cache API, stores the actual data object, so we convert reponse here.
                if (storetype != undefined) {
                    if (storetype.toLowerCase() === "json") {
                        data = yield response.json();
                    }
                    else if (storetype.toLocaleLowerCase() === "arraybuffer") {
                        data = yield response.arrayBuffer();
                    }
                    else {
                        throw Error("Unsupported storetyp for IndexedDB: " + storetype);
                    }
                }
                return new Promise((resolve, reject) => {
                    var _a;
                    const transaction = (_a = this.db) === null || _a === void 0 ? void 0 : _a.transaction(['urls'], 'readwrite');
                    if (transaction === undefined) {
                        return;
                    }
                    const store = transaction.objectStore('urls');
                    const request = store.add({ data, url }); // Index DB follows a {value, key} format, instead of {key, value} format!
                    request.onsuccess = () => resolve();
                    request.onerror = (event) => reject(event.target.error);
                });
            });
        }
        addToCache(url, storetype, signal) {
            return __awaiter(this, void 0, void 0, function* () {
                yield this.initDB(); // await the initDB process
                // If already cached, nothing to do
                const isInDB = yield this.isUrlInDB(url);
                if (isInDB) {
                    return;
                }
                try {
                    const response = yield fetch(url, signal ? { signal } : undefined);
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    const response_copy = response.clone();
                    yield this.addToIndexedDB(url, response_copy, storetype);
                }
                catch (error) {
                    throw Error("Failed to store " + url + " with error: " + error);
                }
            });
        }
        hasAllKeys(keys) {
            return __awaiter(this, void 0, void 0, function* () {
                yield this.initDB(); // Ensure the DB is initialized
                if (!this.db) {
                    throw new Error('Database is not initialized');
                }
                return new Promise((resolve, reject) => {
                    const transaction = this.db.transaction(['urls'], 'readonly');
                    const store = transaction.objectStore('urls');
                    const promises = keys.map(key => {
                        return new Promise((resolve) => {
                            const request = store.get(key);
                            request.onsuccess = () => {
                                if (request.result === undefined) {
                                    resolve(false); // Key not found, resolve with false
                                }
                                else {
                                    resolve(true); // Key found, resolve with true
                                }
                            };
                            request.onerror = () => {
                                resolve(false); // On error, resolve as if the key was not found
                            };
                        });
                    });
                    Promise.all(promises).then(results => {
                        const allExist = results.every(exists => exists);
                        resolve(allExist);
                    }).catch(error => {
                        reject(error); // Reject the main promise if any of the promises are rejected
                    });
                });
            });
        }
        deleteInCache(url) {
            return __awaiter(this, void 0, void 0, function* () {
                var _a;
                yield this.initDB(); // Make sure the DB is initialized
                const transaction = (_a = this.db) === null || _a === void 0 ? void 0 : _a.transaction(['urls'], 'readwrite');
                if (transaction === undefined) {
                    return;
                }
                const store = transaction.objectStore('urls');
                const request = store.delete(url);
                // Await completion of the delete request
                yield new Promise((resolve, reject) => {
                    request.onsuccess = () => resolve();
                    request.onerror = () => reject(request.error);
                });
                return;
            });
        }
    }
    /**
     * Cache by Origin Private File System (OPFS).
     */
    class ArtifactOPFSCache {
        constructor(scope, accessMode = "async") {
            this.store = new OPFSStore(scope, accessMode);
        }
        static isAvailable() {
            return OPFSStore.isAvailable();
        }
        fetchWithCache(url, storetype, signal) {
            return __awaiter(this, void 0, void 0, function* () {
                // TODO: Avoid duplicate OPFS record validation by trying cache reads first
                yield this.addToCache(url, storetype, signal);
                return this.readFromCache(url, storetype);
            });
        }
        readFromCache(url, storetype) {
            return __awaiter(this, void 0, void 0, function* () {
                if ((storetype === null || storetype === void 0 ? void 0 : storetype.toLowerCase()) === "arraybuffer") {
                    const cachedData = yield this.store.readArrayBuffer(url);
                    if (cachedData === undefined) {
                        throw new Error("ArtifactOPFSCache failed to fetch: " + url);
                    }
                    return cachedData;
                }
                const cachedResponse = yield this.store.read(url);
                if (cachedResponse === undefined) {
                    throw new Error("ArtifactOPFSCache failed to fetch: " + url);
                }
                return this.responseToStoreType(cachedResponse, storetype);
            });
        }
        addToCache(url, _storetype, signal) {
            return __awaiter(this, void 0, void 0, function* () {
                if (yield this.store.has(url)) {
                    return;
                }
                const request = new Request(url, signal ? Object.assign(Object.assign({}, DEFAULT_FETCH_OPTIONS), { signal }) : DEFAULT_FETCH_OPTIONS);
                const response = yield fetch(request);
                if (!response.ok) {
                    throw new Error(`ArtifactOPFSCache: Unable to fetch ${url}, received status ${response.status}`);
                }
                yield this.store.write(url, response);
            });
        }
        hasAllKeys(keys) {
            return __awaiter(this, void 0, void 0, function* () {
                const results = yield Promise.all(keys.map((key) => __awaiter(this, void 0, void 0, function* () { return yield this.store.has(key); })));
                return results.every((result) => result);
            });
        }
        deleteInCache(url) {
            return __awaiter(this, void 0, void 0, function* () {
                yield this.store.remove(url);
            });
        }
        responseToStoreType(response, storetype) {
            return __awaiter(this, void 0, void 0, function* () {
                if (storetype === undefined) {
                    return response;
                }
                const format = storetype.toLowerCase();
                if (format === "json") {
                    return response.json();
                }
                if (format === "arraybuffer") {
                    return response.arrayBuffer();
                }
                return response;
            });
        }
    }
    /**
     * Cache by cross-origin storage extension.
     */
    class ArtifactCrossOriginStorageCache {
        constructor(_scope, storage = new CrossOriginStorage()) {
            this.storage = storage;
        }
        fetchWithCache(url, storetype, signal) {
            return __awaiter(this, void 0, void 0, function* () {
                const cachedResponse = yield this.storage.match(url);
                if (cachedResponse !== undefined) {
                    return this.responseToStoreType(cachedResponse, storetype);
                }
                yield this.addToCache(url, storetype, signal);
                const hydrated = yield this.storage.match(url);
                if (hydrated === undefined) {
                    throw new Error(`ArtifactCrossOriginStorageCache: failed to hydrate ${url}`);
                }
                return this.responseToStoreType(hydrated, storetype);
            });
        }
        addToCache(url, _storetype, signal) {
            return __awaiter(this, void 0, void 0, function* () {
                const existing = yield this.storage.match(url);
                if (existing !== undefined) {
                    return;
                }
                const request = new Request(url, signal ? Object.assign(Object.assign({}, DEFAULT_FETCH_OPTIONS), { signal }) : DEFAULT_FETCH_OPTIONS);
                const response = yield fetch(request);
                if (!response.ok) {
                    throw new Error(`ArtifactCrossOriginStorageCache: Unable to fetch ${url}, received status ${response.status}`);
                }
                yield this.storage.put(url, response.clone());
            });
        }
        hasAllKeys(keys) {
            return __awaiter(this, void 0, void 0, function* () {
                const results = yield Promise.all(keys.map((key) => __awaiter(this, void 0, void 0, function* () {
                    const cached = yield this.storage.match(key);
                    return cached !== undefined;
                })));
                return results.every((result) => result);
            });
        }
        deleteInCache(url) {
            return __awaiter(this, void 0, void 0, function* () {
                yield this.storage.delete(url);
            });
        }
        responseToStoreType(response, storetype) {
            return __awaiter(this, void 0, void 0, function* () {
                if (storetype === undefined) {
                    return response;
                }
                const format = storetype.toLowerCase();
                if (format === "json") {
                    return response.json();
                }
                if (format === "arraybuffer") {
                    return response.arrayBuffer();
                }
                return response;
            });
        }
    }
    function normalizeCacheType(cacheType) {
        if (cacheType === undefined) {
            return "cache";
        }
        const normalized = cacheType.toLowerCase();
        if (normalized === "cache") {
            return "cache";
        }
        if (normalized === "indexeddb") {
            return "indexeddb";
        }
        if (normalized === "cross-origin") {
            return "cross-origin";
        }
        if (normalized === "opfs") {
            return "opfs";
        }
        console.error("Unsupported cacheType: " + cacheType + ", using default ArtifactCache.");
        return "cache";
    }
    function isTensorCacheAccessOptions(value) {
        return typeof value === "object" && value !== null;
    }
    function normalizeCacheAccessOptions(cacheScopeOrOptions, cacheType) {
        if (isTensorCacheAccessOptions(cacheScopeOrOptions)) {
            return cacheScopeOrOptions;
        }
        return {
            cacheScope: cacheScopeOrOptions,
            cacheType: normalizeCacheType(cacheType),
        };
    }
    function createArtifactCache(scope, options = {}) {
        if (options.artifactCache !== undefined) {
            return options.artifactCache;
        }
        const cacheType = normalizeCacheType(options.cacheType);
        if (cacheType === "indexeddb") {
            return new ArtifactIndexedDBCache(scope);
        }
        if (cacheType === "cross-origin") {
            if (CrossOriginStorage.isAvailable()) {
                return new ArtifactCrossOriginStorageCache(scope);
            }
            if (!crossOriginFallbackWarningLogged) {
                console.warn("Cross-origin storage backend is unavailable; falling back to ArtifactCache.");
                crossOriginFallbackWarningLogged = true;
            }
        }
        if (cacheType === "opfs") {
            return new ArtifactOPFSCache(scope, options.opfsAccessMode);
        }
        return new ArtifactCache(scope);
    }
    function hasTensorInCache(tensorCacheUrl_1) {
        return __awaiter(this, arguments, void 0, function* (tensorCacheUrl, cacheScopeOrOptions = "tvmjs", cacheType = "cache") {
            var _a;
            const options = normalizeCacheAccessOptions(cacheScopeOrOptions, cacheType);
            const cacheScope = (_a = options.cacheScope) !== null && _a !== void 0 ? _a : "tvmjs";
            const artifactCache = createArtifactCache(cacheScope, options);
            const jsonUrl = new URL("tensor-cache.json", tensorCacheUrl).href;
            const hasJsonUrlInCache = yield artifactCache.hasAllKeys([jsonUrl]);
            if (!hasJsonUrlInCache) {
                return false;
            }
            const list = (yield artifactCache.fetchWithCache(jsonUrl, "json"))["records"];
            return yield artifactCache.hasAllKeys(list.map(key => new URL(key.dataPath, tensorCacheUrl).href));
        });
    }
    function deleteTensorCache(cacheUrl_1) {
        return __awaiter(this, arguments, void 0, function* (cacheUrl, cacheScopeOrOptions = "tvmjs", cacheType = "cache") {
            var _a;
            const options = normalizeCacheAccessOptions(cacheScopeOrOptions, cacheType);
            const cacheScope = (_a = options.cacheScope) !== null && _a !== void 0 ? _a : "tvmjs";
            const artifactCache = createArtifactCache(cacheScope, options);
            if (artifactCache instanceof ArtifactCrossOriginStorageCache) {
                // Cross-origin storage extension does not currently support programmatic deletion.
                return;
            }
            const jsonUrl = new URL("tensor-cache.json", cacheUrl).href;
            const list = yield artifactCache.fetchWithCache(jsonUrl, "json");
            const arrayentry = list["records"];
            const processShard = (i) => __awaiter(this, void 0, void 0, function* () {
                const dataUrl = new URL(arrayentry[i].dataPath, cacheUrl).href;
                yield artifactCache.deleteInCache(dataUrl);
            });
            yield Promise.all(arrayentry.map((_, index) => processShard(index)));
        });
    }

    function EmccWASI() {
    var Module=typeof Module!="undefined"?Module:{};var ENVIRONMENT_IS_WEB=!!globalThis.window;var ENVIRONMENT_IS_WORKER=!!globalThis.WorkerGlobalScope;var ENVIRONMENT_IS_NODE=globalThis.process?.versions?.node&&globalThis.process?.type!="renderer";var __wasmLib={};function __wasmLibInstantiateWasm(imports,successCallback){__wasmLib.imports=imports;__wasmLib.successCallback=successCallback;}function __wasmLibStart(wasmInstance){__wasmLib.successCallback(wasmInstance);}__wasmLib.start=__wasmLibStart;var Module={instantiateWasm:__wasmLibInstantiateWasm,wasmLibraryProvider:__wasmLib};var arguments_=[];var thisProgram="./this.program";var quit_=(status,toThrow)=>{throw toThrow};var _scriptName=globalThis.document?.currentScript?.src;if(typeof __filename!="undefined"){_scriptName=__filename;}else if(ENVIRONMENT_IS_WORKER){_scriptName=self.location.href;}var scriptDirectory="";function locateFile(path){if(Module["locateFile"]){return Module["locateFile"](path,scriptDirectory)}return scriptDirectory+path}var readAsync,readBinary;if(ENVIRONMENT_IS_NODE){var fs=require("node:fs");scriptDirectory=__dirname+"/";readBinary=filename=>{filename=isFileURI(filename)?new URL(filename):filename;var ret=fs.readFileSync(filename);return ret};readAsync=async(filename,binary=true)=>{filename=isFileURI(filename)?new URL(filename):filename;var ret=fs.readFileSync(filename,binary?undefined:"utf8");return ret};if(process.argv.length>1){thisProgram=process.argv[1].replace(/\\/g,"/");}arguments_=process.argv.slice(2);if(typeof module!="undefined"){module["exports"]=Module;}quit_=(status,toThrow)=>{process.exitCode=status;throw toThrow};}else if(ENVIRONMENT_IS_WEB||ENVIRONMENT_IS_WORKER){try{scriptDirectory=new URL(".",_scriptName).href;}catch{}{if(ENVIRONMENT_IS_WORKER){readBinary=url=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,false);xhr.responseType="arraybuffer";xhr.send(null);return new Uint8Array(xhr.response)};}readAsync=async url=>{if(isFileURI(url)){return new Promise((resolve,reject)=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,true);xhr.responseType="arraybuffer";xhr.onload=()=>{if(xhr.status==200||xhr.status==0&&xhr.response){resolve(xhr.response);return}reject(xhr.status);};xhr.onerror=reject;xhr.send(null);})}var response=await fetch(url,{credentials:"same-origin"});if(response.ok){return response.arrayBuffer()}throw new Error(response.status+" : "+response.url)};}}else;var out=console.log.bind(console);var err=console.error.bind(console);var wasmBinary;var ABORT=false;var EXITSTATUS;var isFileURI=filename=>filename.startsWith("file://");function updateMemoryViews(){var b=wasmMemory.buffer;HEAP8=new Int8Array(b);HEAPU8=new Uint8Array(b);HEAP32=new Int32Array(b);HEAPU32=new Uint32Array(b);HEAP64=new BigInt64Array(b);new BigUint64Array(b);}function preRun(){if(Module["preRun"]){if(typeof Module["preRun"]=="function")Module["preRun"]=[Module["preRun"]];while(Module["preRun"].length){addOnPreRun(Module["preRun"].shift());}}callRuntimeCallbacks(onPreRuns);}function initRuntime(){if(!Module["noFSInit"]&&!FS.initialized)FS.init();FS.ignorePermissions=false;}function postRun(){if(Module["postRun"]){if(typeof Module["postRun"]=="function")Module["postRun"]=[Module["postRun"]];while(Module["postRun"].length){addOnPostRun(Module["postRun"].shift());}}callRuntimeCallbacks(onPostRuns);}function abort(what){Module["onAbort"]?.(what);what=`Aborted(${what})`;err(what);ABORT=true;what+=". Build with -sASSERTIONS for more info.";var e=new WebAssembly.RuntimeError(what);throw e}var wasmBinaryFile;function findWasmBinary(){return locateFile("tvmjs_runtime.wasm")}function getBinarySync(file){if(file==wasmBinaryFile&&wasmBinary){return new Uint8Array(wasmBinary)}if(readBinary){return readBinary(file)}throw "both async and sync fetching of the wasm failed"}async function getWasmBinary(binaryFile){if(!wasmBinary){try{var response=await readAsync(binaryFile);return new Uint8Array(response)}catch{}}return getBinarySync(binaryFile)}async function instantiateArrayBuffer(binaryFile,imports){try{var binary=await getWasmBinary(binaryFile);var instance=await WebAssembly.instantiate(binary,imports);return instance}catch(reason){err(`failed to asynchronously prepare wasm: ${reason}`);abort(reason);}}async function instantiateAsync(binary,binaryFile,imports){if(!binary&&!isFileURI(binaryFile)&&!ENVIRONMENT_IS_NODE){try{var response=fetch(binaryFile,{credentials:"same-origin"});var instantiationResult=await WebAssembly.instantiateStreaming(response,imports);return instantiationResult}catch(reason){err(`wasm streaming compile failed: ${reason}`);err("falling back to ArrayBuffer instantiation");}}return instantiateArrayBuffer(binaryFile,imports)}function getWasmImports(){var imports={env:wasmImports,wasi_snapshot_preview1:wasmImports};return imports}async function createWasm(){function receiveInstance(instance,module){wasmExports=instance.exports;wasmExports=Asyncify.instrumentWasmExports(wasmExports);assignWasmExports(wasmExports);updateMemoryViews();removeRunDependency();return wasmExports}addRunDependency();function receiveInstantiationResult(result){return receiveInstance(result["instance"])}var info=getWasmImports();if(Module["instantiateWasm"]){return new Promise((resolve,reject)=>{Module["instantiateWasm"](info,(inst,mod)=>{resolve(receiveInstance(inst));});})}wasmBinaryFile??=findWasmBinary();var result=await instantiateAsync(wasmBinary,wasmBinaryFile,info);var exports=receiveInstantiationResult(result);return exports}class ExitStatus{name="ExitStatus";constructor(status){this.message=`Program terminated with exit(${status})`;this.status=status;}}var HEAP32;var HEAP64;var HEAP8;var HEAPU32;var HEAPU8;var callRuntimeCallbacks=callbacks=>{while(callbacks.length>0){callbacks.shift()(Module);}};var onPostRuns=[];var addOnPostRun=cb=>onPostRuns.push(cb);var onPreRuns=[];var addOnPreRun=cb=>onPreRuns.push(cb);var runDependencies=0;var dependenciesFulfilled=null;var removeRunDependency=id=>{runDependencies--;Module["monitorRunDependencies"]?.(runDependencies);if(runDependencies==0){if(dependenciesFulfilled){var callback=dependenciesFulfilled;dependenciesFulfilled=null;callback();}}};var addRunDependency=id=>{runDependencies++;Module["monitorRunDependencies"]?.(runDependencies);};var noExitRuntime=true;function _TVMFFIWasmFunctionDeleter(...args){abort("missing function: TVMFFIWasmFunctionDeleter");}_TVMFFIWasmFunctionDeleter.stub=true;function _TVMFFIWasmSafeCall(...args){abort("missing function: TVMFFIWasmSafeCall");}_TVMFFIWasmSafeCall.stub=true;var _emscripten_get_now=()=>performance.now();var _emscripten_date_now=()=>Date.now();var checkWasiClock=clock_id=>clock_id>=0&&clock_id<=3;var INT53_MAX=9007199254740992;var INT53_MIN=-9007199254740992;var bigintToI53Checked=num=>num<INT53_MIN||num>INT53_MAX?NaN:Number(num);function _clock_time_get(clk_id,ignored_precision,ptime){if(!checkWasiClock(clk_id)){return 28}var now;if(clk_id===0){now=_emscripten_date_now();}else {now=_emscripten_get_now();}var nsec=Math.round(now*1e3*1e3);HEAP64[ptime>>3]=BigInt(nsec);return 0}var _emscripten_notify_memory_growth=memoryIndex=>{updateMemoryViews();};var ENV={};var getExecutableName=()=>thisProgram||"./this.program";var getEnvStrings=()=>{if(!getEnvStrings.strings){var lang=(globalThis.navigator?.language??"C").replace("-","_")+".UTF-8";var env={USER:"web_user",LOGNAME:"web_user",PATH:"/",PWD:"/",HOME:"/home/web_user",LANG:lang,_:getExecutableName()};for(var x in ENV){if(ENV[x]===undefined)delete env[x];else env[x]=ENV[x];}var strings=[];for(var x in env){strings.push(`${x}=${env[x]}`);}getEnvStrings.strings=strings;}return getEnvStrings.strings};var stringToUTF8Array=(str,heap,outIdx,maxBytesToWrite)=>{if(!(maxBytesToWrite>0))return 0;var startIdx=outIdx;var endIdx=outIdx+maxBytesToWrite-1;for(var i=0;i<str.length;++i){var u=str.codePointAt(i);if(u<=127){if(outIdx>=endIdx)break;heap[outIdx++]=u;}else if(u<=2047){if(outIdx+1>=endIdx)break;heap[outIdx++]=192|u>>6;heap[outIdx++]=128|u&63;}else if(u<=65535){if(outIdx+2>=endIdx)break;heap[outIdx++]=224|u>>12;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63;}else {if(outIdx+3>=endIdx)break;heap[outIdx++]=240|u>>18;heap[outIdx++]=128|u>>12&63;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63;i++;}}heap[outIdx]=0;return outIdx-startIdx};var stringToUTF8=(str,outPtr,maxBytesToWrite)=>stringToUTF8Array(str,HEAPU8,outPtr,maxBytesToWrite);var _environ_get=(__environ,environ_buf)=>{var bufSize=0;var envp=0;for(var string of getEnvStrings()){var ptr=environ_buf+bufSize;HEAPU32[__environ+envp>>2]=ptr;bufSize+=stringToUTF8(string,ptr,Infinity)+1;envp+=4;}return 0};var lengthBytesUTF8=str=>{var len=0;for(var i=0;i<str.length;++i){var c=str.charCodeAt(i);if(c<=127){len++;}else if(c<=2047){len+=2;}else if(c>=55296&&c<=57343){len+=4;++i;}else {len+=3;}}return len};var _environ_sizes_get=(penviron_count,penviron_buf_size)=>{var strings=getEnvStrings();HEAPU32[penviron_count>>2]=strings.length;var bufSize=0;for(var string of strings){bufSize+=lengthBytesUTF8(string)+1;}HEAPU32[penviron_buf_size>>2]=bufSize;return 0};var PATH={isAbs:path=>path.charAt(0)==="/",splitPath:filename=>{var splitPathRe=/^(\/?|)([\s\S]*?)((?:\.{1,2}|[^\/]+?|)(\.[^.\/]*|))(?:[\/]*)$/;return splitPathRe.exec(filename).slice(1)},normalizeArray:(parts,allowAboveRoot)=>{var up=0;for(var i=parts.length-1;i>=0;i--){var last=parts[i];if(last==="."){parts.splice(i,1);}else if(last===".."){parts.splice(i,1);up++;}else if(up){parts.splice(i,1);up--;}}if(allowAboveRoot){for(;up;up--){parts.unshift("..");}}return parts},normalize:path=>{var isAbsolute=PATH.isAbs(path),trailingSlash=path.slice(-1)==="/";path=PATH.normalizeArray(path.split("/").filter(p=>!!p),!isAbsolute).join("/");if(!path&&!isAbsolute){path=".";}if(path&&trailingSlash){path+="/";}return (isAbsolute?"/":"")+path},dirname:path=>{var result=PATH.splitPath(path),root=result[0],dir=result[1];if(!root&&!dir){return "."}if(dir){dir=dir.slice(0,-1);}return root+dir},basename:path=>path&&path.match(/([^\/]+|\/)\/*$/)[1],join:(...paths)=>PATH.normalize(paths.join("/")),join2:(l,r)=>PATH.normalize(l+"/"+r)};var initRandomFill=()=>{if(ENVIRONMENT_IS_NODE){var nodeCrypto=require("node:crypto");return view=>nodeCrypto.randomFillSync(view)}return view=>(crypto.getRandomValues(view),0)};var randomFill=view=>(randomFill=initRandomFill())(view);var PATH_FS={resolve:(...args)=>{var resolvedPath="",resolvedAbsolute=false;for(var i=args.length-1;i>=-1&&!resolvedAbsolute;i--){var path=i>=0?args[i]:FS.cwd();if(typeof path!="string"){throw new TypeError("Arguments to path.resolve must be strings")}else if(!path){return ""}resolvedPath=path+"/"+resolvedPath;resolvedAbsolute=PATH.isAbs(path);}resolvedPath=PATH.normalizeArray(resolvedPath.split("/").filter(p=>!!p),!resolvedAbsolute).join("/");return (resolvedAbsolute?"/":"")+resolvedPath||"."},relative:(from,to)=>{from=PATH_FS.resolve(from).slice(1);to=PATH_FS.resolve(to).slice(1);function trim(arr){var start=0;for(;start<arr.length;start++){if(arr[start]!=="")break}var end=arr.length-1;for(;end>=0;end--){if(arr[end]!=="")break}if(start>end)return [];return arr.slice(start,end-start+1)}var fromParts=trim(from.split("/"));var toParts=trim(to.split("/"));var length=Math.min(fromParts.length,toParts.length);var samePartsLength=length;for(var i=0;i<length;i++){if(fromParts[i]!==toParts[i]){samePartsLength=i;break}}var outputParts=[];for(var i=samePartsLength;i<fromParts.length;i++){outputParts.push("..");}outputParts=outputParts.concat(toParts.slice(samePartsLength));return outputParts.join("/")}};var UTF8Decoder=globalThis.TextDecoder&&new TextDecoder;var findStringEnd=(heapOrArray,idx,maxBytesToRead,ignoreNul)=>{var maxIdx=idx+maxBytesToRead;while(heapOrArray[idx]&&!(idx>=maxIdx))++idx;return idx};var UTF8ArrayToString=(heapOrArray,idx=0,maxBytesToRead,ignoreNul)=>{var endPtr=findStringEnd(heapOrArray,idx,maxBytesToRead);if(endPtr-idx>16&&heapOrArray.buffer&&UTF8Decoder){return UTF8Decoder.decode(heapOrArray.subarray(idx,endPtr))}var str="";while(idx<endPtr){var u0=heapOrArray[idx++];if(!(u0&128)){str+=String.fromCharCode(u0);continue}var u1=heapOrArray[idx++]&63;if((u0&224)==192){str+=String.fromCharCode((u0&31)<<6|u1);continue}var u2=heapOrArray[idx++]&63;if((u0&240)==224){u0=(u0&15)<<12|u1<<6|u2;}else {u0=(u0&7)<<18|u1<<12|u2<<6|heapOrArray[idx++]&63;}if(u0<65536){str+=String.fromCharCode(u0);}else {var ch=u0-65536;str+=String.fromCharCode(55296|ch>>10,56320|ch&1023);}}return str};var FS_stdin_getChar_buffer=[];var intArrayFromString=(stringy,dontAddNull,length)=>{var len=lengthBytesUTF8(stringy)+1;var u8array=new Array(len);var numBytesWritten=stringToUTF8Array(stringy,u8array,0,u8array.length);u8array.length=numBytesWritten;return u8array};var FS_stdin_getChar=()=>{if(!FS_stdin_getChar_buffer.length){var result=null;if(ENVIRONMENT_IS_NODE){var BUFSIZE=256;var buf=Buffer.alloc(BUFSIZE);var bytesRead=0;var fd=process.stdin.fd;try{bytesRead=fs.readSync(fd,buf,0,BUFSIZE);}catch(e){if(e.toString().includes("EOF"))bytesRead=0;else throw e}if(bytesRead>0){result=buf.slice(0,bytesRead).toString("utf-8");}}else if(globalThis.window?.prompt){result=window.prompt("Input: ");if(result!==null){result+="\n";}}else;if(!result){return null}FS_stdin_getChar_buffer=intArrayFromString(result);}return FS_stdin_getChar_buffer.shift()};var TTY={ttys:[],init(){},shutdown(){},register(dev,ops){TTY.ttys[dev]={input:[],output:[],ops};FS.registerDevice(dev,TTY.stream_ops);},stream_ops:{open(stream){var tty=TTY.ttys[stream.node.rdev];if(!tty){throw new FS.ErrnoError(43)}stream.tty=tty;stream.seekable=false;},close(stream){stream.tty.ops.fsync(stream.tty);},fsync(stream){stream.tty.ops.fsync(stream.tty);},read(stream,buffer,offset,length,pos){if(!stream.tty||!stream.tty.ops.get_char){throw new FS.ErrnoError(60)}var bytesRead=0;for(var i=0;i<length;i++){var result;try{result=stream.tty.ops.get_char(stream.tty);}catch(e){throw new FS.ErrnoError(29)}if(result===undefined&&bytesRead===0){throw new FS.ErrnoError(6)}if(result===null||result===undefined)break;bytesRead++;buffer[offset+i]=result;}if(bytesRead){stream.node.atime=Date.now();}return bytesRead},write(stream,buffer,offset,length,pos){if(!stream.tty||!stream.tty.ops.put_char){throw new FS.ErrnoError(60)}try{for(var i=0;i<length;i++){stream.tty.ops.put_char(stream.tty,buffer[offset+i]);}}catch(e){throw new FS.ErrnoError(29)}if(length){stream.node.mtime=stream.node.ctime=Date.now();}return i}},default_tty_ops:{get_char(tty){return FS_stdin_getChar()},put_char(tty,val){if(val===null||val===10){out(UTF8ArrayToString(tty.output));tty.output=[];}else {if(val!=0)tty.output.push(val);}},fsync(tty){if(tty.output?.length>0){out(UTF8ArrayToString(tty.output));tty.output=[];}},ioctl_tcgets(tty){return {c_iflag:25856,c_oflag:5,c_cflag:191,c_lflag:35387,c_cc:[3,28,127,21,4,0,1,0,17,19,26,0,18,15,23,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}},ioctl_tcsets(tty,optional_actions,data){return 0},ioctl_tiocgwinsz(tty){return [24,80]}},default_tty1_ops:{put_char(tty,val){if(val===null||val===10){err(UTF8ArrayToString(tty.output));tty.output=[];}else {if(val!=0)tty.output.push(val);}},fsync(tty){if(tty.output?.length>0){err(UTF8ArrayToString(tty.output));tty.output=[];}}}};var mmapAlloc=size=>{abort();};var MEMFS={ops_table:null,mount(mount){return MEMFS.createNode(null,"/",16895,0)},createNode(parent,name,mode,dev){if(FS.isBlkdev(mode)||FS.isFIFO(mode)){throw new FS.ErrnoError(63)}MEMFS.ops_table||={dir:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr,lookup:MEMFS.node_ops.lookup,mknod:MEMFS.node_ops.mknod,rename:MEMFS.node_ops.rename,unlink:MEMFS.node_ops.unlink,rmdir:MEMFS.node_ops.rmdir,readdir:MEMFS.node_ops.readdir,symlink:MEMFS.node_ops.symlink},stream:{llseek:MEMFS.stream_ops.llseek}},file:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr},stream:{llseek:MEMFS.stream_ops.llseek,read:MEMFS.stream_ops.read,write:MEMFS.stream_ops.write,mmap:MEMFS.stream_ops.mmap,msync:MEMFS.stream_ops.msync}},link:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr,readlink:MEMFS.node_ops.readlink},stream:{}},chrdev:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr},stream:FS.chrdev_stream_ops}};var node=FS.createNode(parent,name,mode,dev);if(FS.isDir(node.mode)){node.node_ops=MEMFS.ops_table.dir.node;node.stream_ops=MEMFS.ops_table.dir.stream;node.contents={};}else if(FS.isFile(node.mode)){node.node_ops=MEMFS.ops_table.file.node;node.stream_ops=MEMFS.ops_table.file.stream;node.usedBytes=0;node.contents=MEMFS.emptyFileContents??=new Uint8Array(0);}else if(FS.isLink(node.mode)){node.node_ops=MEMFS.ops_table.link.node;node.stream_ops=MEMFS.ops_table.link.stream;}else if(FS.isChrdev(node.mode)){node.node_ops=MEMFS.ops_table.chrdev.node;node.stream_ops=MEMFS.ops_table.chrdev.stream;}node.atime=node.mtime=node.ctime=Date.now();if(parent){parent.contents[name]=node;parent.atime=parent.mtime=parent.ctime=node.atime;}return node},getFileDataAsTypedArray(node){return node.contents.subarray(0,node.usedBytes)},expandFileStorage(node,newCapacity){var prevCapacity=node.contents.length;if(prevCapacity>=newCapacity)return;var CAPACITY_DOUBLING_MAX=1024*1024;newCapacity=Math.max(newCapacity,prevCapacity*(prevCapacity<CAPACITY_DOUBLING_MAX?2:1.125)>>>0);if(prevCapacity)newCapacity=Math.max(newCapacity,256);var oldContents=MEMFS.getFileDataAsTypedArray(node);node.contents=new Uint8Array(newCapacity);node.contents.set(oldContents);},resizeFileStorage(node,newSize){if(node.usedBytes==newSize)return;var oldContents=node.contents;node.contents=new Uint8Array(newSize);node.contents.set(oldContents.subarray(0,Math.min(newSize,node.usedBytes)));node.usedBytes=newSize;},node_ops:{getattr(node){var attr={};attr.dev=FS.isChrdev(node.mode)?node.id:1;attr.ino=node.id;attr.mode=node.mode;attr.nlink=1;attr.uid=0;attr.gid=0;attr.rdev=node.rdev;if(FS.isDir(node.mode)){attr.size=4096;}else if(FS.isFile(node.mode)){attr.size=node.usedBytes;}else if(FS.isLink(node.mode)){attr.size=node.link.length;}else {attr.size=0;}attr.atime=new Date(node.atime);attr.mtime=new Date(node.mtime);attr.ctime=new Date(node.ctime);attr.blksize=4096;attr.blocks=Math.ceil(attr.size/attr.blksize);return attr},setattr(node,attr){for(const key of ["mode","atime","mtime","ctime"]){if(attr[key]!=null){node[key]=attr[key];}}if(attr.size!==undefined){MEMFS.resizeFileStorage(node,attr.size);}},lookup(parent,name){if(!MEMFS.doesNotExistError){MEMFS.doesNotExistError=new FS.ErrnoError(44);MEMFS.doesNotExistError.stack="<generic error, no stack>";}throw MEMFS.doesNotExistError},mknod(parent,name,mode,dev){return MEMFS.createNode(parent,name,mode,dev)},rename(old_node,new_dir,new_name){var new_node;try{new_node=FS.lookupNode(new_dir,new_name);}catch(e){}if(new_node){if(FS.isDir(old_node.mode)){for(var i in new_node.contents){throw new FS.ErrnoError(55)}}FS.hashRemoveNode(new_node);}delete old_node.parent.contents[old_node.name];new_dir.contents[new_name]=old_node;old_node.name=new_name;new_dir.ctime=new_dir.mtime=old_node.parent.ctime=old_node.parent.mtime=Date.now();},unlink(parent,name){delete parent.contents[name];parent.ctime=parent.mtime=Date.now();},rmdir(parent,name){var node=FS.lookupNode(parent,name);for(var i in node.contents){throw new FS.ErrnoError(55)}delete parent.contents[name];parent.ctime=parent.mtime=Date.now();},readdir(node){return [".","..",...Object.keys(node.contents)]},symlink(parent,newname,oldpath){var node=MEMFS.createNode(parent,newname,511|40960,0);node.link=oldpath;return node},readlink(node){if(!FS.isLink(node.mode)){throw new FS.ErrnoError(28)}return node.link}},stream_ops:{read(stream,buffer,offset,length,position){var contents=stream.node.contents;if(position>=stream.node.usedBytes)return 0;var size=Math.min(stream.node.usedBytes-position,length);buffer.set(contents.subarray(position,position+size),offset);return size},write(stream,buffer,offset,length,position,canOwn){if(buffer.buffer===HEAP8.buffer){canOwn=false;}if(!length)return 0;var node=stream.node;node.mtime=node.ctime=Date.now();if(canOwn){node.contents=buffer.subarray(offset,offset+length);node.usedBytes=length;}else if(node.usedBytes===0&&position===0){node.contents=buffer.slice(offset,offset+length);node.usedBytes=length;}else {MEMFS.expandFileStorage(node,position+length);node.contents.set(buffer.subarray(offset,offset+length),position);node.usedBytes=Math.max(node.usedBytes,position+length);}return length},llseek(stream,offset,whence){var position=offset;if(whence===1){position+=stream.position;}else if(whence===2){if(FS.isFile(stream.node.mode)){position+=stream.node.usedBytes;}}if(position<0){throw new FS.ErrnoError(28)}return position},mmap(stream,length,position,prot,flags){if(!FS.isFile(stream.node.mode)){throw new FS.ErrnoError(43)}var ptr;var allocated;var contents=stream.node.contents;if(!(flags&2)&&contents.buffer===HEAP8.buffer){allocated=false;ptr=contents.byteOffset;}else {allocated=true;ptr=mmapAlloc();if(!ptr){throw new FS.ErrnoError(48)}if(contents){if(position>0||position+length<contents.length){if(contents.subarray){contents=contents.subarray(position,position+length);}else {contents=Array.prototype.slice.call(contents,position,position+length);}}HEAP8.set(contents,ptr);}}return {ptr,allocated}},msync(stream,buffer,offset,length,mmapFlags){MEMFS.stream_ops.write(stream,buffer,0,length,offset,false);return 0}}};var FS_modeStringToFlags=str=>{if(typeof str!="string")return str;var flagModes={r:0,"r+":2,w:512|64|1,"w+":512|64|2,a:1024|64|1,"a+":1024|64|2};var flags=flagModes[str];if(typeof flags=="undefined"){throw new Error(`Unknown file open mode: ${str}`)}return flags};var FS_fileDataToTypedArray=data=>{if(typeof data=="string"){data=intArrayFromString(data);}if(!data.subarray){data=new Uint8Array(data);}return data};var FS_getMode=(canRead,canWrite)=>{var mode=0;if(canRead)mode|=292|73;if(canWrite)mode|=146;return mode};var asyncLoad=async url=>{var arrayBuffer=await readAsync(url);return new Uint8Array(arrayBuffer)};var FS_createDataFile=(...args)=>FS.createDataFile(...args);var preloadPlugins=[];var FS_handledByPreloadPlugin=async(byteArray,fullname)=>{if(typeof Browser!="undefined")Browser.init();for(var plugin of preloadPlugins){if(plugin["canHandle"](fullname)){return plugin["handle"](byteArray,fullname)}}return byteArray};var FS_preloadFile=async(parent,name,url,canRead,canWrite,dontCreateFile,canOwn,preFinish)=>{var fullname=name?PATH_FS.resolve(PATH.join2(parent,name)):parent;addRunDependency();try{var byteArray=url;if(typeof url=="string"){byteArray=await asyncLoad(url);}byteArray=await FS_handledByPreloadPlugin(byteArray,fullname);preFinish?.();if(!dontCreateFile){FS_createDataFile(parent,name,byteArray,canRead,canWrite,canOwn);}}finally{removeRunDependency();}};var FS_createPreloadedFile=(parent,name,url,canRead,canWrite,onload,onerror,dontCreateFile,canOwn,preFinish)=>{FS_preloadFile(parent,name,url,canRead,canWrite,dontCreateFile,canOwn,preFinish).then(onload).catch(onerror);};var FS={root:null,mounts:[],devices:{},streams:[],nextInode:1,nameTable:null,currentPath:"/",initialized:false,ignorePermissions:true,filesystems:null,syncFSRequests:0,ErrnoError:class{name="ErrnoError";constructor(errno){this.errno=errno;}},FSStream:class{shared={};get object(){return this.node}set object(val){this.node=val;}get isRead(){return (this.flags&2097155)!==1}get isWrite(){return (this.flags&2097155)!==0}get isAppend(){return this.flags&1024}get flags(){return this.shared.flags}set flags(val){this.shared.flags=val;}get position(){return this.shared.position}set position(val){this.shared.position=val;}},FSNode:class{node_ops={};stream_ops={};readMode=292|73;writeMode=146;mounted=null;constructor(parent,name,mode,rdev){if(!parent){parent=this;}this.parent=parent;this.mount=parent.mount;this.id=FS.nextInode++;this.name=name;this.mode=mode;this.rdev=rdev;this.atime=this.mtime=this.ctime=Date.now();}get read(){return (this.mode&this.readMode)===this.readMode}set read(val){val?this.mode|=this.readMode:this.mode&=~this.readMode;}get write(){return (this.mode&this.writeMode)===this.writeMode}set write(val){val?this.mode|=this.writeMode:this.mode&=~this.writeMode;}get isFolder(){return FS.isDir(this.mode)}get isDevice(){return FS.isChrdev(this.mode)}},lookupPath(path,opts={}){if(!path){throw new FS.ErrnoError(44)}opts.follow_mount??=true;if(!PATH.isAbs(path)){path=FS.cwd()+"/"+path;}linkloop:for(var nlinks=0;nlinks<40;nlinks++){var parts=path.split("/").filter(p=>!!p);var current=FS.root;var current_path="/";for(var i=0;i<parts.length;i++){var islast=i===parts.length-1;if(islast&&opts.parent){break}if(parts[i]==="."){continue}if(parts[i]===".."){current_path=PATH.dirname(current_path);if(FS.isRoot(current)){path=current_path+"/"+parts.slice(i+1).join("/");nlinks--;continue linkloop}else {current=current.parent;}continue}current_path=PATH.join2(current_path,parts[i]);try{current=FS.lookupNode(current,parts[i]);}catch(e){if(e?.errno===44&&islast&&opts.noent_okay){return {path:current_path}}throw e}if(FS.isMountpoint(current)&&(!islast||opts.follow_mount)){current=current.mounted.root;}if(FS.isLink(current.mode)&&(!islast||opts.follow)){if(!current.node_ops.readlink){throw new FS.ErrnoError(52)}var link=current.node_ops.readlink(current);if(!PATH.isAbs(link)){link=PATH.dirname(current_path)+"/"+link;}path=link+"/"+parts.slice(i+1).join("/");continue linkloop}}return {path:current_path,node:current}}throw new FS.ErrnoError(32)},getPath(node){var path;while(true){if(FS.isRoot(node)){var mount=node.mount.mountpoint;if(!path)return mount;return mount[mount.length-1]!=="/"?`${mount}/${path}`:mount+path}path=path?`${node.name}/${path}`:node.name;node=node.parent;}},hashName(parentid,name){var hash=0;for(var i=0;i<name.length;i++){hash=(hash<<5)-hash+name.charCodeAt(i)|0;}return (parentid+hash>>>0)%FS.nameTable.length},hashAddNode(node){var hash=FS.hashName(node.parent.id,node.name);node.name_next=FS.nameTable[hash];FS.nameTable[hash]=node;},hashRemoveNode(node){var hash=FS.hashName(node.parent.id,node.name);if(FS.nameTable[hash]===node){FS.nameTable[hash]=node.name_next;}else {var current=FS.nameTable[hash];while(current){if(current.name_next===node){current.name_next=node.name_next;break}current=current.name_next;}}},lookupNode(parent,name){var errCode=FS.mayLookup(parent);if(errCode){throw new FS.ErrnoError(errCode)}var hash=FS.hashName(parent.id,name);for(var node=FS.nameTable[hash];node;node=node.name_next){var nodeName=node.name;if(node.parent.id===parent.id&&nodeName===name){return node}}return FS.lookup(parent,name)},createNode(parent,name,mode,rdev){var node=new FS.FSNode(parent,name,mode,rdev);FS.hashAddNode(node);return node},destroyNode(node){FS.hashRemoveNode(node);},isRoot(node){return node===node.parent},isMountpoint(node){return !!node.mounted},isFile(mode){return (mode&61440)===32768},isDir(mode){return (mode&61440)===16384},isLink(mode){return (mode&61440)===40960},isChrdev(mode){return (mode&61440)===8192},isBlkdev(mode){return (mode&61440)===24576},isFIFO(mode){return (mode&61440)===4096},isSocket(mode){return (mode&49152)===49152},flagsToPermissionString(flag){var perms=["r","w","rw"][flag&3];if(flag&512){perms+="w";}return perms},nodePermissions(node,perms){if(FS.ignorePermissions){return 0}if(perms.includes("r")&&!(node.mode&292)){return 2}if(perms.includes("w")&&!(node.mode&146)){return 2}if(perms.includes("x")&&!(node.mode&73)){return 2}return 0},mayLookup(dir){if(!FS.isDir(dir.mode))return 54;var errCode=FS.nodePermissions(dir,"x");if(errCode)return errCode;if(!dir.node_ops.lookup)return 2;return 0},mayCreate(dir,name){if(!FS.isDir(dir.mode)){return 54}try{var node=FS.lookupNode(dir,name);return 20}catch(e){}return FS.nodePermissions(dir,"wx")},mayDelete(dir,name,isdir){var node;try{node=FS.lookupNode(dir,name);}catch(e){return e.errno}var errCode=FS.nodePermissions(dir,"wx");if(errCode){return errCode}if(isdir){if(!FS.isDir(node.mode)){return 54}if(FS.isRoot(node)||FS.getPath(node)===FS.cwd()){return 10}}else if(FS.isDir(node.mode)){return 31}return 0},mayOpen(node,flags){if(!node){return 44}if(FS.isLink(node.mode)){return 32}var mode=FS.flagsToPermissionString(flags);if(FS.isDir(node.mode)){if(mode!=="r"||flags&(512|64)){return 31}}return FS.nodePermissions(node,mode)},checkOpExists(op,err){if(!op){throw new FS.ErrnoError(err)}return op},MAX_OPEN_FDS:4096,nextfd(){for(var fd=0;fd<=FS.MAX_OPEN_FDS;fd++){if(!FS.streams[fd]){return fd}}throw new FS.ErrnoError(33)},getStreamChecked(fd){var stream=FS.getStream(fd);if(!stream){throw new FS.ErrnoError(8)}return stream},getStream:fd=>FS.streams[fd],createStream(stream,fd=-1){stream=Object.assign(new FS.FSStream,stream);if(fd==-1){fd=FS.nextfd();}stream.fd=fd;FS.streams[fd]=stream;return stream},closeStream(fd){FS.streams[fd]=null;},dupStream(origStream,fd=-1){var stream=FS.createStream(origStream,fd);stream.stream_ops?.dup?.(stream);return stream},doSetAttr(stream,node,attr){var setattr=stream?.stream_ops.setattr;var arg=setattr?stream:node;setattr??=node.node_ops.setattr;FS.checkOpExists(setattr,63);try{setattr(arg,attr);}catch(e){if(e instanceof RangeError){throw new FS.ErrnoError(22)}throw e}},chrdev_stream_ops:{open(stream){var device=FS.getDevice(stream.node.rdev);stream.stream_ops=device.stream_ops;stream.stream_ops.open?.(stream);},llseek(){throw new FS.ErrnoError(70)}},major:dev=>dev>>8,minor:dev=>dev&255,makedev:(ma,mi)=>ma<<8|mi,registerDevice(dev,ops){FS.devices[dev]={stream_ops:ops};},getDevice:dev=>FS.devices[dev],getMounts(mount){var mounts=[];var check=[mount];while(check.length){var m=check.pop();mounts.push(m);check.push(...m.mounts);}return mounts},syncfs(populate,callback){if(typeof populate=="function"){callback=populate;populate=false;}FS.syncFSRequests++;if(FS.syncFSRequests>1){err(`warning: ${FS.syncFSRequests} FS.syncfs operations in flight at once, probably just doing extra work`);}var mounts=FS.getMounts(FS.root.mount);var completed=0;function doCallback(errCode){FS.syncFSRequests--;return callback(errCode)}function done(errCode){if(errCode){if(!done.errored){done.errored=true;return doCallback(errCode)}return}if(++completed>=mounts.length){doCallback(null);}}for(var mount of mounts){if(mount.type.syncfs){mount.type.syncfs(mount,populate,done);}else {done(null);}}},mount(type,opts,mountpoint){var root=mountpoint==="/";var pseudo=!mountpoint;var node;if(root&&FS.root){throw new FS.ErrnoError(10)}else if(!root&&!pseudo){var lookup=FS.lookupPath(mountpoint,{follow_mount:false});mountpoint=lookup.path;node=lookup.node;if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}if(!FS.isDir(node.mode)){throw new FS.ErrnoError(54)}}var mount={type,opts,mountpoint,mounts:[]};var mountRoot=type.mount(mount);mountRoot.mount=mount;mount.root=mountRoot;if(root){FS.root=mountRoot;}else if(node){node.mounted=mount;if(node.mount){node.mount.mounts.push(mount);}}return mountRoot},unmount(mountpoint){var lookup=FS.lookupPath(mountpoint,{follow_mount:false});if(!FS.isMountpoint(lookup.node)){throw new FS.ErrnoError(28)}var node=lookup.node;var mount=node.mounted;var mounts=FS.getMounts(mount);for(var[hash,current]of Object.entries(FS.nameTable)){while(current){var next=current.name_next;if(mounts.includes(current.mount)){FS.destroyNode(current);}current=next;}}node.mounted=null;var idx=node.mount.mounts.indexOf(mount);node.mount.mounts.splice(idx,1);},lookup(parent,name){return parent.node_ops.lookup(parent,name)},mknod(path,mode,dev){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;var name=PATH.basename(path);if(!name){throw new FS.ErrnoError(28)}if(name==="."||name===".."){throw new FS.ErrnoError(20)}var errCode=FS.mayCreate(parent,name);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.mknod){throw new FS.ErrnoError(63)}return parent.node_ops.mknod(parent,name,mode,dev)},statfs(path){return FS.statfsNode(FS.lookupPath(path,{follow:true}).node)},statfsStream(stream){return FS.statfsNode(stream.node)},statfsNode(node){var rtn={bsize:4096,frsize:4096,blocks:1e6,bfree:5e5,bavail:5e5,files:FS.nextInode,ffree:FS.nextInode-1,fsid:42,flags:2,namelen:255};if(node.node_ops.statfs){Object.assign(rtn,node.node_ops.statfs(node.mount.opts.root));}return rtn},create(path,mode=438){mode&=4095;mode|=32768;return FS.mknod(path,mode,0)},mkdir(path,mode=511){mode&=511|512;mode|=16384;return FS.mknod(path,mode,0)},mkdirTree(path,mode){var dirs=path.split("/");var d="";for(var dir of dirs){if(!dir)continue;if(d||PATH.isAbs(path))d+="/";d+=dir;try{FS.mkdir(d,mode);}catch(e){if(e.errno!=20)throw e}}},mkdev(path,mode,dev){if(typeof dev=="undefined"){dev=mode;mode=438;}mode|=8192;return FS.mknod(path,mode,dev)},symlink(oldpath,newpath){if(!PATH_FS.resolve(oldpath)){throw new FS.ErrnoError(44)}var lookup=FS.lookupPath(newpath,{parent:true});var parent=lookup.node;if(!parent){throw new FS.ErrnoError(44)}var newname=PATH.basename(newpath);var errCode=FS.mayCreate(parent,newname);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.symlink){throw new FS.ErrnoError(63)}return parent.node_ops.symlink(parent,newname,oldpath)},rename(old_path,new_path){var old_dirname=PATH.dirname(old_path);var new_dirname=PATH.dirname(new_path);var old_name=PATH.basename(old_path);var new_name=PATH.basename(new_path);var lookup,old_dir,new_dir;lookup=FS.lookupPath(old_path,{parent:true});old_dir=lookup.node;lookup=FS.lookupPath(new_path,{parent:true});new_dir=lookup.node;if(!old_dir||!new_dir)throw new FS.ErrnoError(44);if(old_dir.mount!==new_dir.mount){throw new FS.ErrnoError(75)}var old_node=FS.lookupNode(old_dir,old_name);var relative=PATH_FS.relative(old_path,new_dirname);if(relative.charAt(0)!=="."){throw new FS.ErrnoError(28)}relative=PATH_FS.relative(new_path,old_dirname);if(relative.charAt(0)!=="."){throw new FS.ErrnoError(55)}var new_node;try{new_node=FS.lookupNode(new_dir,new_name);}catch(e){}if(old_node===new_node){return}var isdir=FS.isDir(old_node.mode);var errCode=FS.mayDelete(old_dir,old_name,isdir);if(errCode){throw new FS.ErrnoError(errCode)}errCode=new_node?FS.mayDelete(new_dir,new_name,isdir):FS.mayCreate(new_dir,new_name);if(errCode){throw new FS.ErrnoError(errCode)}if(!old_dir.node_ops.rename){throw new FS.ErrnoError(63)}if(FS.isMountpoint(old_node)||new_node&&FS.isMountpoint(new_node)){throw new FS.ErrnoError(10)}if(new_dir!==old_dir){errCode=FS.nodePermissions(old_dir,"w");if(errCode){throw new FS.ErrnoError(errCode)}}FS.hashRemoveNode(old_node);try{old_dir.node_ops.rename(old_node,new_dir,new_name);old_node.parent=new_dir;}catch(e){throw e}finally{FS.hashAddNode(old_node);}},rmdir(path){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;var name=PATH.basename(path);var node=FS.lookupNode(parent,name);var errCode=FS.mayDelete(parent,name,true);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.rmdir){throw new FS.ErrnoError(63)}if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}parent.node_ops.rmdir(parent,name);FS.destroyNode(node);},readdir(path){var lookup=FS.lookupPath(path,{follow:true});var node=lookup.node;var readdir=FS.checkOpExists(node.node_ops.readdir,54);return readdir(node)},unlink(path){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;if(!parent){throw new FS.ErrnoError(44)}var name=PATH.basename(path);var node=FS.lookupNode(parent,name);var errCode=FS.mayDelete(parent,name,false);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.unlink){throw new FS.ErrnoError(63)}if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}parent.node_ops.unlink(parent,name);FS.destroyNode(node);},readlink(path){var lookup=FS.lookupPath(path);var link=lookup.node;if(!link){throw new FS.ErrnoError(44)}if(!link.node_ops.readlink){throw new FS.ErrnoError(28)}return link.node_ops.readlink(link)},stat(path,dontFollow){var lookup=FS.lookupPath(path,{follow:!dontFollow});var node=lookup.node;var getattr=FS.checkOpExists(node.node_ops.getattr,63);return getattr(node)},fstat(fd){var stream=FS.getStreamChecked(fd);var node=stream.node;var getattr=stream.stream_ops.getattr;var arg=getattr?stream:node;getattr??=node.node_ops.getattr;FS.checkOpExists(getattr,63);return getattr(arg)},lstat(path){return FS.stat(path,true)},doChmod(stream,node,mode,dontFollow){FS.doSetAttr(stream,node,{mode:mode&4095|node.mode&-4096,ctime:Date.now(),dontFollow});},chmod(path,mode,dontFollow){var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:!dontFollow});node=lookup.node;}else {node=path;}FS.doChmod(null,node,mode,dontFollow);},lchmod(path,mode){FS.chmod(path,mode,true);},fchmod(fd,mode){var stream=FS.getStreamChecked(fd);FS.doChmod(stream,stream.node,mode,false);},doChown(stream,node,dontFollow){FS.doSetAttr(stream,node,{timestamp:Date.now(),dontFollow});},chown(path,uid,gid,dontFollow){var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:!dontFollow});node=lookup.node;}else {node=path;}FS.doChown(null,node,dontFollow);},lchown(path,uid,gid){FS.chown(path,uid,gid,true);},fchown(fd,uid,gid){var stream=FS.getStreamChecked(fd);FS.doChown(stream,stream.node,false);},doTruncate(stream,node,len){if(FS.isDir(node.mode)){throw new FS.ErrnoError(31)}if(!FS.isFile(node.mode)){throw new FS.ErrnoError(28)}var errCode=FS.nodePermissions(node,"w");if(errCode){throw new FS.ErrnoError(errCode)}FS.doSetAttr(stream,node,{size:len,timestamp:Date.now()});},truncate(path,len){if(len<0){throw new FS.ErrnoError(28)}var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:true});node=lookup.node;}else {node=path;}FS.doTruncate(null,node,len);},ftruncate(fd,len){var stream=FS.getStreamChecked(fd);if(len<0||(stream.flags&2097155)===0){throw new FS.ErrnoError(28)}FS.doTruncate(stream,stream.node,len);},utime(path,atime,mtime){var lookup=FS.lookupPath(path,{follow:true});var node=lookup.node;var setattr=FS.checkOpExists(node.node_ops.setattr,63);setattr(node,{atime,mtime});},open(path,flags,mode=438){if(path===""){throw new FS.ErrnoError(44)}flags=FS_modeStringToFlags(flags);if(flags&64){mode=mode&4095|32768;}else {mode=0;}var node;var isDirPath;if(typeof path=="object"){node=path;}else {isDirPath=path.endsWith("/");var lookup=FS.lookupPath(path,{follow:!(flags&131072),noent_okay:true});node=lookup.node;path=lookup.path;}var created=false;if(flags&64){if(node){if(flags&128){throw new FS.ErrnoError(20)}}else if(isDirPath){throw new FS.ErrnoError(31)}else {node=FS.mknod(path,mode|511,0);created=true;}}if(!node){throw new FS.ErrnoError(44)}if(FS.isChrdev(node.mode)){flags&=-513;}if(flags&65536&&!FS.isDir(node.mode)){throw new FS.ErrnoError(54)}if(!created){var errCode=FS.mayOpen(node,flags);if(errCode){throw new FS.ErrnoError(errCode)}}if(flags&512&&!created){FS.truncate(node,0);}flags&=-131713;var stream=FS.createStream({node,path:FS.getPath(node),flags,seekable:true,position:0,stream_ops:node.stream_ops,ungotten:[],error:false});if(stream.stream_ops.open){stream.stream_ops.open(stream);}if(created){FS.chmod(node,mode&511);}return stream},close(stream){if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if(stream.getdents)stream.getdents=null;try{if(stream.stream_ops.close){stream.stream_ops.close(stream);}}catch(e){throw e}finally{FS.closeStream(stream.fd);}stream.fd=null;},isClosed(stream){return stream.fd===null},llseek(stream,offset,whence){if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if(!stream.seekable||!stream.stream_ops.llseek){throw new FS.ErrnoError(70)}if(whence!=0&&whence!=1&&whence!=2){throw new FS.ErrnoError(28)}stream.position=stream.stream_ops.llseek(stream,offset,whence);stream.ungotten=[];return stream.position},read(stream,buffer,offset,length,position){if(length<0||position<0){throw new FS.ErrnoError(28)}if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if((stream.flags&2097155)===1){throw new FS.ErrnoError(8)}if(FS.isDir(stream.node.mode)){throw new FS.ErrnoError(31)}if(!stream.stream_ops.read){throw new FS.ErrnoError(28)}var seeking=typeof position!="undefined";if(!seeking){position=stream.position;}else if(!stream.seekable){throw new FS.ErrnoError(70)}var bytesRead=stream.stream_ops.read(stream,buffer,offset,length,position);if(!seeking)stream.position+=bytesRead;return bytesRead},write(stream,buffer,offset,length,position,canOwn){if(length<0||position<0){throw new FS.ErrnoError(28)}if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if((stream.flags&2097155)===0){throw new FS.ErrnoError(8)}if(FS.isDir(stream.node.mode)){throw new FS.ErrnoError(31)}if(!stream.stream_ops.write){throw new FS.ErrnoError(28)}if(stream.seekable&&stream.flags&1024){FS.llseek(stream,0,2);}var seeking=typeof position!="undefined";if(!seeking){position=stream.position;}else if(!stream.seekable){throw new FS.ErrnoError(70)}var bytesWritten=stream.stream_ops.write(stream,buffer,offset,length,position,canOwn);if(!seeking)stream.position+=bytesWritten;return bytesWritten},mmap(stream,length,position,prot,flags){if((prot&2)!==0&&(flags&2)===0&&(stream.flags&2097155)!==2){throw new FS.ErrnoError(2)}if((stream.flags&2097155)===1){throw new FS.ErrnoError(2)}if(!stream.stream_ops.mmap){throw new FS.ErrnoError(43)}if(!length){throw new FS.ErrnoError(28)}return stream.stream_ops.mmap(stream,length,position,prot,flags)},msync(stream,buffer,offset,length,mmapFlags){if(!stream.stream_ops.msync){return 0}return stream.stream_ops.msync(stream,buffer,offset,length,mmapFlags)},ioctl(stream,cmd,arg){if(!stream.stream_ops.ioctl){throw new FS.ErrnoError(59)}return stream.stream_ops.ioctl(stream,cmd,arg)},readFile(path,opts={}){opts.flags=opts.flags||0;opts.encoding=opts.encoding||"binary";if(opts.encoding!=="utf8"&&opts.encoding!=="binary"){abort(`Invalid encoding type "${opts.encoding}"`);}var stream=FS.open(path,opts.flags);var stat=FS.stat(path);var length=stat.size;var buf=new Uint8Array(length);FS.read(stream,buf,0,length,0);if(opts.encoding==="utf8"){buf=UTF8ArrayToString(buf);}FS.close(stream);return buf},writeFile(path,data,opts={}){opts.flags=opts.flags||577;var stream=FS.open(path,opts.flags,opts.mode);data=FS_fileDataToTypedArray(data);FS.write(stream,data,0,data.byteLength,undefined,opts.canOwn);FS.close(stream);},cwd:()=>FS.currentPath,chdir(path){var lookup=FS.lookupPath(path,{follow:true});if(lookup.node===null){throw new FS.ErrnoError(44)}if(!FS.isDir(lookup.node.mode)){throw new FS.ErrnoError(54)}var errCode=FS.nodePermissions(lookup.node,"x");if(errCode){throw new FS.ErrnoError(errCode)}FS.currentPath=lookup.path;},createDefaultDirectories(){FS.mkdir("/tmp");FS.mkdir("/home");FS.mkdir("/home/web_user");},createDefaultDevices(){FS.mkdir("/dev");FS.registerDevice(FS.makedev(1,3),{read:()=>0,write:(stream,buffer,offset,length,pos)=>length,llseek:()=>0});FS.mkdev("/dev/null",FS.makedev(1,3));TTY.register(FS.makedev(5,0),TTY.default_tty_ops);TTY.register(FS.makedev(6,0),TTY.default_tty1_ops);FS.mkdev("/dev/tty",FS.makedev(5,0));FS.mkdev("/dev/tty1",FS.makedev(6,0));var randomBuffer=new Uint8Array(1024),randomLeft=0;var randomByte=()=>{if(randomLeft===0){randomFill(randomBuffer);randomLeft=randomBuffer.byteLength;}return randomBuffer[--randomLeft]};FS.createDevice("/dev","random",randomByte);FS.createDevice("/dev","urandom",randomByte);FS.mkdir("/dev/shm");FS.mkdir("/dev/shm/tmp");},createSpecialDirectories(){FS.mkdir("/proc");var proc_self=FS.mkdir("/proc/self");FS.mkdir("/proc/self/fd");FS.mount({mount(){var node=FS.createNode(proc_self,"fd",16895,73);node.stream_ops={llseek:MEMFS.stream_ops.llseek};node.node_ops={lookup(parent,name){var fd=+name;var stream=FS.getStreamChecked(fd);var ret={parent:null,mount:{mountpoint:"fake"},node_ops:{readlink:()=>stream.path},id:fd+1};ret.parent=ret;return ret},readdir(){return Array.from(FS.streams.entries()).filter(([k,v])=>v).map(([k,v])=>k.toString())}};return node}},{},"/proc/self/fd");},createStandardStreams(input,output,error){if(input){FS.createDevice("/dev","stdin",input);}else {FS.symlink("/dev/tty","/dev/stdin");}if(output){FS.createDevice("/dev","stdout",null,output);}else {FS.symlink("/dev/tty","/dev/stdout");}if(error){FS.createDevice("/dev","stderr",null,error);}else {FS.symlink("/dev/tty1","/dev/stderr");}FS.open("/dev/stdin",0);FS.open("/dev/stdout",1);FS.open("/dev/stderr",1);},staticInit(){FS.nameTable=new Array(4096);FS.mount(MEMFS,{},"/");FS.createDefaultDirectories();FS.createDefaultDevices();FS.createSpecialDirectories();FS.filesystems={MEMFS};},init(input,output,error){FS.initialized=true;input??=Module["stdin"];output??=Module["stdout"];error??=Module["stderr"];FS.createStandardStreams(input,output,error);},quit(){FS.initialized=false;for(var stream of FS.streams){if(stream){FS.close(stream);}}},findObject(path,dontResolveLastLink){var ret=FS.analyzePath(path,dontResolveLastLink);if(!ret.exists){return null}return ret.object},analyzePath(path,dontResolveLastLink){try{var lookup=FS.lookupPath(path,{follow:!dontResolveLastLink});path=lookup.path;}catch(e){}var ret={isRoot:false,exists:false,error:0,name:null,path:null,object:null,parentExists:false,parentPath:null,parentObject:null};try{var lookup=FS.lookupPath(path,{parent:true});ret.parentExists=true;ret.parentPath=lookup.path;ret.parentObject=lookup.node;ret.name=PATH.basename(path);lookup=FS.lookupPath(path,{follow:!dontResolveLastLink});ret.exists=true;ret.path=lookup.path;ret.object=lookup.node;ret.name=lookup.node.name;ret.isRoot=lookup.path==="/";}catch(e){ret.error=e.errno;}return ret},createPath(parent,path,canRead,canWrite){parent=typeof parent=="string"?parent:FS.getPath(parent);var parts=path.split("/").reverse();while(parts.length){var part=parts.pop();if(!part)continue;var current=PATH.join2(parent,part);try{FS.mkdir(current);}catch(e){if(e.errno!=20)throw e}parent=current;}return current},createFile(parent,name,properties,canRead,canWrite){var path=PATH.join2(typeof parent=="string"?parent:FS.getPath(parent),name);var mode=FS_getMode(canRead,canWrite);return FS.create(path,mode)},createDataFile(parent,name,data,canRead,canWrite,canOwn){var path=name;if(parent){parent=typeof parent=="string"?parent:FS.getPath(parent);path=name?PATH.join2(parent,name):parent;}var mode=FS_getMode(canRead,canWrite);var node=FS.create(path,mode);if(data){data=FS_fileDataToTypedArray(data);FS.chmod(node,mode|146);var stream=FS.open(node,577);FS.write(stream,data,0,data.length,0,canOwn);FS.close(stream);FS.chmod(node,mode);}},createDevice(parent,name,input,output){var path=PATH.join2(typeof parent=="string"?parent:FS.getPath(parent),name);var mode=FS_getMode(!!input,!!output);FS.createDevice.major??=64;var dev=FS.makedev(FS.createDevice.major++,0);FS.registerDevice(dev,{open(stream){stream.seekable=false;},close(stream){if(output?.buffer?.length){output(10);}},read(stream,buffer,offset,length,pos){var bytesRead=0;for(var i=0;i<length;i++){var result;try{result=input();}catch(e){throw new FS.ErrnoError(29)}if(result===undefined&&bytesRead===0){throw new FS.ErrnoError(6)}if(result===null||result===undefined)break;bytesRead++;buffer[offset+i]=result;}if(bytesRead){stream.node.atime=Date.now();}return bytesRead},write(stream,buffer,offset,length,pos){for(var i=0;i<length;i++){try{output(buffer[offset+i]);}catch(e){throw new FS.ErrnoError(29)}}if(length){stream.node.mtime=stream.node.ctime=Date.now();}return i}});return FS.mkdev(path,mode,dev)},forceLoadFile(obj){if(obj.isDevice||obj.isFolder||obj.link||obj.contents)return true;if(globalThis.XMLHttpRequest){abort("Lazy loading should have been performed (contents set) in createLazyFile, but it was not. Lazy loading only works in web workers. Use --embed-file or --preload-file in emcc on the main thread.");}else {try{obj.contents=readBinary(obj.url);}catch(e){throw new FS.ErrnoError(29)}}},createLazyFile(parent,name,url,canRead,canWrite){class LazyUint8Array{lengthKnown=false;chunks=[];get(idx){if(idx>this.length-1||idx<0){return undefined}var chunkOffset=idx%this.chunkSize;var chunkNum=idx/this.chunkSize|0;return this.getter(chunkNum)[chunkOffset]}setDataGetter(getter){this.getter=getter;}cacheLength(){var xhr=new XMLHttpRequest;xhr.open("HEAD",url,false);xhr.send(null);if(!(xhr.status>=200&&xhr.status<300||xhr.status===304))abort("Couldn't load "+url+". Status: "+xhr.status);var datalength=Number(xhr.getResponseHeader("Content-length"));var header;var hasByteServing=(header=xhr.getResponseHeader("Accept-Ranges"))&&header==="bytes";var usesGzip=(header=xhr.getResponseHeader("Content-Encoding"))&&header==="gzip";var chunkSize=1024*1024;if(!hasByteServing)chunkSize=datalength;var doXHR=(from,to)=>{if(from>to)abort("invalid range ("+from+", "+to+") or no bytes requested!");if(to>datalength-1)abort("only "+datalength+" bytes available! programmer error!");var xhr=new XMLHttpRequest;xhr.open("GET",url,false);if(datalength!==chunkSize)xhr.setRequestHeader("Range","bytes="+from+"-"+to);xhr.responseType="arraybuffer";if(xhr.overrideMimeType){xhr.overrideMimeType("text/plain; charset=x-user-defined");}xhr.send(null);if(!(xhr.status>=200&&xhr.status<300||xhr.status===304))abort("Couldn't load "+url+". Status: "+xhr.status);if(xhr.response!==undefined){return new Uint8Array(xhr.response||[])}return intArrayFromString(xhr.responseText||"")};var lazyArray=this;lazyArray.setDataGetter(chunkNum=>{var start=chunkNum*chunkSize;var end=(chunkNum+1)*chunkSize-1;end=Math.min(end,datalength-1);if(typeof lazyArray.chunks[chunkNum]=="undefined"){lazyArray.chunks[chunkNum]=doXHR(start,end);}if(typeof lazyArray.chunks[chunkNum]=="undefined")abort("doXHR failed!");return lazyArray.chunks[chunkNum]});if(usesGzip||!datalength){chunkSize=datalength=1;datalength=this.getter(0).length;chunkSize=datalength;out("LazyFiles on gzip forces download of the whole file when length is accessed");}this._length=datalength;this._chunkSize=chunkSize;this.lengthKnown=true;}get length(){if(!this.lengthKnown){this.cacheLength();}return this._length}get chunkSize(){if(!this.lengthKnown){this.cacheLength();}return this._chunkSize}}if(globalThis.XMLHttpRequest){if(!ENVIRONMENT_IS_WORKER)abort("Cannot do synchronous binary XHRs outside webworkers in modern browsers. Use --embed-file or --preload-file in emcc");var lazyArray=new LazyUint8Array;var properties={isDevice:false,contents:lazyArray};}else {var properties={isDevice:false,url};}var node=FS.createFile(parent,name,properties,canRead,canWrite);if(properties.contents){node.contents=properties.contents;}else if(properties.url){node.contents=null;node.url=properties.url;}Object.defineProperties(node,{usedBytes:{get:function(){return this.contents.length}}});var stream_ops={};for(const[key,fn]of Object.entries(node.stream_ops)){stream_ops[key]=(...args)=>{FS.forceLoadFile(node);return fn(...args)};}function writeChunks(stream,buffer,offset,length,position){var contents=stream.node.contents;if(position>=contents.length)return 0;var size=Math.min(contents.length-position,length);if(contents.slice){for(var i=0;i<size;i++){buffer[offset+i]=contents[position+i];}}else {for(var i=0;i<size;i++){buffer[offset+i]=contents.get(position+i);}}return size}stream_ops.read=(stream,buffer,offset,length,position)=>{FS.forceLoadFile(node);return writeChunks(stream,buffer,offset,length,position)};stream_ops.mmap=(stream,length,position,prot,flags)=>{FS.forceLoadFile(node);var ptr=mmapAlloc();if(!ptr){throw new FS.ErrnoError(48)}writeChunks(stream,HEAP8,ptr,length,position);return {ptr,allocated:true}};node.stream_ops=stream_ops;return node}};var UTF8ToString=(ptr,maxBytesToRead,ignoreNul)=>ptr?UTF8ArrayToString(HEAPU8,ptr,maxBytesToRead):"";var SYSCALLS={currentUmask:18,calculateAt(dirfd,path,allowEmpty){if(PATH.isAbs(path)){return path}var dir;if(dirfd===-100){dir=FS.cwd();}else {var dirstream=SYSCALLS.getStreamFromFD(dirfd);dir=dirstream.path;}if(path.length==0){if(!allowEmpty){throw new FS.ErrnoError(44)}return dir}return dir+"/"+path},writeStat(buf,stat){HEAPU32[buf>>2]=stat.dev;HEAPU32[buf+4>>2]=stat.mode;HEAPU32[buf+8>>2]=stat.nlink;HEAPU32[buf+12>>2]=stat.uid;HEAPU32[buf+16>>2]=stat.gid;HEAPU32[buf+20>>2]=stat.rdev;HEAP64[buf+24>>3]=BigInt(stat.size);HEAP32[buf+32>>2]=4096;HEAP32[buf+36>>2]=stat.blocks;var atime=stat.atime.getTime();var mtime=stat.mtime.getTime();var ctime=stat.ctime.getTime();HEAP64[buf+40>>3]=BigInt(Math.floor(atime/1e3));HEAPU32[buf+48>>2]=atime%1e3*1e3*1e3;HEAP64[buf+56>>3]=BigInt(Math.floor(mtime/1e3));HEAPU32[buf+64>>2]=mtime%1e3*1e3*1e3;HEAP64[buf+72>>3]=BigInt(Math.floor(ctime/1e3));HEAPU32[buf+80>>2]=ctime%1e3*1e3*1e3;HEAP64[buf+88>>3]=BigInt(stat.ino);return 0},writeStatFs(buf,stats){HEAPU32[buf+4>>2]=stats.bsize;HEAPU32[buf+60>>2]=stats.bsize;HEAP64[buf+8>>3]=BigInt(stats.blocks);HEAP64[buf+16>>3]=BigInt(stats.bfree);HEAP64[buf+24>>3]=BigInt(stats.bavail);HEAP64[buf+32>>3]=BigInt(stats.files);HEAP64[buf+40>>3]=BigInt(stats.ffree);HEAPU32[buf+48>>2]=stats.fsid;HEAPU32[buf+64>>2]=stats.flags;HEAPU32[buf+56>>2]=stats.namelen;},doMsync(addr,stream,len,flags,offset){if(!FS.isFile(stream.node.mode)){throw new FS.ErrnoError(43)}if(flags&2){return 0}var buffer=HEAPU8.slice(addr,addr+len);FS.msync(stream,buffer,offset,len,flags);},getStreamFromFD(fd){var stream=FS.getStreamChecked(fd);return stream},varargs:undefined,getStr(ptr){var ret=UTF8ToString(ptr);return ret}};function _fd_close(fd){try{var stream=SYSCALLS.getStreamFromFD(fd);FS.close(stream);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var doReadv=(stream,iov,iovcnt,offset)=>{var ret=0;for(var i=0;i<iovcnt;i++){var ptr=HEAPU32[iov>>2];var len=HEAPU32[iov+4>>2];iov+=8;var curr=FS.read(stream,HEAP8,ptr,len,offset);if(curr<0)return  -1;ret+=curr;if(curr<len)break;}return ret};function _fd_read(fd,iov,iovcnt,pnum){try{var stream=SYSCALLS.getStreamFromFD(fd);var num=doReadv(stream,iov,iovcnt);HEAPU32[pnum>>2]=num;return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _fd_seek(fd,offset,whence,newOffset){offset=bigintToI53Checked(offset);try{if(isNaN(offset))return 22;var stream=SYSCALLS.getStreamFromFD(fd);FS.llseek(stream,offset,whence);HEAP64[newOffset>>3]=BigInt(stream.position);if(stream.getdents&&offset===0&&whence===0)stream.getdents=null;return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var doWritev=(stream,iov,iovcnt,offset)=>{var ret=0;for(var i=0;i<iovcnt;i++){var ptr=HEAPU32[iov>>2];var len=HEAPU32[iov+4>>2];iov+=8;var curr=FS.write(stream,HEAP8,ptr,len,offset);if(curr<0)return  -1;ret+=curr;if(curr<len){break}}return ret};function _fd_write(fd,iov,iovcnt,pnum){try{var stream=SYSCALLS.getStreamFromFD(fd);var num=doWritev(stream,iov,iovcnt);HEAPU32[pnum>>2]=num;return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var runtimeKeepaliveCounter=0;var keepRuntimeAlive=()=>noExitRuntime||runtimeKeepaliveCounter>0;var _proc_exit=code=>{EXITSTATUS=code;if(!keepRuntimeAlive()){Module["onExit"]?.(code);ABORT=true;}quit_(code,new ExitStatus(code));};var exitJS=(status,implicit)=>{EXITSTATUS=status;_proc_exit(status);};var handleException=e=>{if(e instanceof ExitStatus||e=="unwind"){return EXITSTATUS}quit_(1,e);};var runAndAbortIfError=func=>{try{return func()}catch(e){abort(e);}};var _exit=exitJS;var maybeExit=()=>{if(!keepRuntimeAlive()){try{_exit(EXITSTATUS);}catch(e){handleException(e);}}};var callUserCallback=func=>{if(ABORT){return}try{return func()}catch(e){handleException(e);}finally{maybeExit();}};var Asyncify={instrumentWasmImports(imports){var importPattern=/^(invoke_.*|__asyncjs__.*)$/;for(let[x,original]of Object.entries(imports)){if(typeof original=="function"){original.isAsync||importPattern.test(x);}}},instrumentFunction(original){var wrapper=(...args)=>{Asyncify.exportCallStack.push(original);try{return original(...args)}finally{if(!ABORT){Asyncify.exportCallStack.pop();Asyncify.maybeStopUnwind();}}};Asyncify.funcWrappers.set(original,wrapper);return wrapper},instrumentWasmExports(exports){var ret={};for(let[x,original]of Object.entries(exports)){if(typeof original=="function"){var wrapper=Asyncify.instrumentFunction(original);ret[x]=wrapper;}else {ret[x]=original;}}return ret},State:{Normal:0,Unwinding:1,Rewinding:2,Disabled:3},state:0,StackSize:4096,currData:null,handleSleepReturnValue:0,exportCallStack:[],callstackFuncToId:new Map,callStackIdToFunc:new Map,funcWrappers:new Map,callStackId:0,asyncPromiseHandlers:null,sleepCallbacks:[],getCallStackId(func){if(!Asyncify.callstackFuncToId.has(func)){var id=Asyncify.callStackId++;Asyncify.callstackFuncToId.set(func,id);Asyncify.callStackIdToFunc.set(id,func);}return Asyncify.callstackFuncToId.get(func)},maybeStopUnwind(){if(Asyncify.currData&&Asyncify.state===Asyncify.State.Unwinding&&Asyncify.exportCallStack.length===0){Asyncify.state=Asyncify.State.Normal;runAndAbortIfError(_asyncify_stop_unwind);if(typeof Fibers!="undefined"){Fibers.trampoline();}}},whenDone(){return new Promise((resolve,reject)=>{Asyncify.asyncPromiseHandlers={resolve,reject};})},allocateData(){var ptr=_malloc(12+Asyncify.StackSize);Asyncify.setDataHeader(ptr,ptr+12,Asyncify.StackSize);Asyncify.setDataRewindFunc(ptr);return ptr},setDataHeader(ptr,stack,stackSize){HEAPU32[ptr>>2]=stack;HEAPU32[ptr+4>>2]=stack+stackSize;},setDataRewindFunc(ptr){var bottomOfCallStack=Asyncify.exportCallStack[0];var rewindId=Asyncify.getCallStackId(bottomOfCallStack);HEAP32[ptr+8>>2]=rewindId;},getDataRewindFunc(ptr){var id=HEAP32[ptr+8>>2];var func=Asyncify.callStackIdToFunc.get(id);return func},doRewind(ptr){var original=Asyncify.getDataRewindFunc(ptr);var func=Asyncify.funcWrappers.get(original);return callUserCallback(func)},handleSleep(startAsync){if(ABORT)return;if(Asyncify.state===Asyncify.State.Normal){var reachedCallback=false;var reachedAfterCallback=false;startAsync((handleSleepReturnValue=0)=>{if(ABORT)return;Asyncify.handleSleepReturnValue=handleSleepReturnValue;reachedCallback=true;if(!reachedAfterCallback){return}Asyncify.state=Asyncify.State.Rewinding;runAndAbortIfError(()=>_asyncify_start_rewind(Asyncify.currData));if(typeof MainLoop!="undefined"&&MainLoop.func){MainLoop.resume();}var asyncWasmReturnValue,isError=false;try{asyncWasmReturnValue=Asyncify.doRewind(Asyncify.currData);}catch(err){asyncWasmReturnValue=err;isError=true;}var handled=false;if(!Asyncify.currData){var asyncPromiseHandlers=Asyncify.asyncPromiseHandlers;if(asyncPromiseHandlers){Asyncify.asyncPromiseHandlers=null;(isError?asyncPromiseHandlers.reject:asyncPromiseHandlers.resolve)(asyncWasmReturnValue);handled=true;}}if(isError&&!handled){throw asyncWasmReturnValue}});reachedAfterCallback=true;if(!reachedCallback){Asyncify.state=Asyncify.State.Unwinding;Asyncify.currData=Asyncify.allocateData();if(typeof MainLoop!="undefined"&&MainLoop.func){MainLoop.pause();}runAndAbortIfError(()=>_asyncify_start_unwind(Asyncify.currData));}}else if(Asyncify.state===Asyncify.State.Rewinding){Asyncify.state=Asyncify.State.Normal;runAndAbortIfError(_asyncify_stop_rewind);_free(Asyncify.currData);Asyncify.currData=null;Asyncify.sleepCallbacks.forEach(callUserCallback);}else {abort(`invalid state: ${Asyncify.state}`);}return Asyncify.handleSleepReturnValue},handleAsync:startAsync=>Asyncify.handleSleep(async wakeUp=>{wakeUp(await startAsync());})};FS.createPreloadedFile=FS_createPreloadedFile;FS.preloadFile=FS_preloadFile;FS.staticInit();{if(Module["noExitRuntime"])noExitRuntime=Module["noExitRuntime"];if(Module["preloadPlugins"])preloadPlugins=Module["preloadPlugins"];if(Module["print"])out=Module["print"];if(Module["printErr"])err=Module["printErr"];if(Module["wasmBinary"])wasmBinary=Module["wasmBinary"];if(Module["arguments"])arguments_=Module["arguments"];if(Module["thisProgram"])thisProgram=Module["thisProgram"];if(Module["preInit"]){if(typeof Module["preInit"]=="function")Module["preInit"]=[Module["preInit"]];while(Module["preInit"].length>0){Module["preInit"].shift()();}}}var _malloc,_free,__initialize,_asyncify_start_unwind,_asyncify_stop_unwind,_asyncify_start_rewind,_asyncify_stop_rewind,wasmMemory;function assignWasmExports(wasmExports){Module["_TVMFFIBacktrace"]=wasmExports["TVMFFIBacktrace"];_malloc=wasmExports["malloc"];Module["_TVMFFIEnvSetStream"]=wasmExports["TVMFFIEnvSetStream"];Module["_TVMFFIEnvGetStream"]=wasmExports["TVMFFIEnvGetStream"];Module["_TVMBackendGetFuncFromEnv"]=wasmExports["TVMBackendGetFuncFromEnv"];Module["_TVMFFIEnvModLookupFromImports"]=wasmExports["TVMFFIEnvModLookupFromImports"];Module["_TVMBackendAllocWorkspace"]=wasmExports["TVMBackendAllocWorkspace"];Module["_TVMBackendFreeWorkspace"]=wasmExports["TVMBackendFreeWorkspace"];Module["_TVMBackendRunOnce"]=wasmExports["TVMBackendRunOnce"];Module["__ZN3tvm7runtime13GetFileFormatERKNSt3__212basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEES9_"]=wasmExports["_ZN3tvm7runtime13GetFileFormatERKNSt3__212basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEES9_"];Module["__ZN3tvm7runtime15GetMetaFilePathERKNSt3__212basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEE"]=wasmExports["_ZN3tvm7runtime15GetMetaFilePathERKNSt3__212basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEE"];Module["__ZN3tvm7runtime18LoadBinaryFromFileERKNSt3__212basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEEPS7_"]=wasmExports["_ZN3tvm7runtime18LoadBinaryFromFileERKNSt3__212basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEEPS7_"];Module["__ZN3tvm7runtime16SaveBinaryToFileERKNSt3__212basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEES9_"]=wasmExports["_ZN3tvm7runtime16SaveBinaryToFileERKNSt3__212basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEES9_"];Module["__ZN3tvm7runtime18SaveMetaDataToFileERKNSt3__212basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEERKNS_3ffi3MapINSA_6StringENS0_12FunctionInfoEvEE"]=wasmExports["_ZN3tvm7runtime18SaveMetaDataToFileERKNSt3__212basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEERKNS_3ffi3MapINSA_6StringENS0_12FunctionInfoEvEE"];Module["_TVMFFIDataTypeToString"]=wasmExports["TVMFFIDataTypeToString"];Module["__ZN3tvm3ffi4json9StringifyERKNS0_3AnyENS0_8OptionalIivEE"]=wasmExports["_ZN3tvm3ffi4json9StringifyERKNS0_3AnyENS0_8OptionalIivEE"];Module["__ZN3tvm7runtime20LoadMetaDataFromFileERKNSt3__212basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEEPNS_3ffi3MapINSA_6StringENS0_12FunctionInfoEvEE"]=wasmExports["_ZN3tvm7runtime20LoadMetaDataFromFileERKNSt3__212basic_stringIcNS1_11char_traitsIcEENS1_9allocatorIcEEEEPNS_3ffi3MapINSA_6StringENS0_12FunctionInfoEvEE"];Module["__ZN3tvm3ffi4json5ParseERKNS0_6StringEPS2_"]=wasmExports["_ZN3tvm3ffi4json5ParseERKNS0_6StringEPS2_"];Module["__ZN3tvm7runtime6Tensor5EmptyENS_3ffi5ShapeE10DLDataType8DLDeviceNS2_8OptionalINS2_6StringEvEE"]=wasmExports["_ZN3tvm7runtime6Tensor5EmptyENS_3ffi5ShapeE10DLDataType8DLDeviceNS2_8OptionalINS2_6StringEvEE"];Module["__ZN3tvm7runtime6Tensor11CopyToBytesEPK8DLTensorPvmS5_"]=wasmExports["_ZN3tvm7runtime6Tensor11CopyToBytesEPK8DLTensorPvmS5_"];Module["_TVMFFITypeGetOrAllocIndex"]=wasmExports["TVMFFITypeGetOrAllocIndex"];Module["_TVMFFIFunctionGetGlobal"]=wasmExports["TVMFFIFunctionGetGlobal"];Module["__ZN3tvm7runtime6Tensor13CopyFromBytesEPK8DLTensorPvmS5_"]=wasmExports["_ZN3tvm7runtime6Tensor13CopyFromBytesEPK8DLTensorPvmS5_"];Module["__ZNK3tvm7runtime6Tensor10CreateViewENS_3ffi5ShapeE10DLDataTypey"]=wasmExports["_ZNK3tvm7runtime6Tensor10CreateViewENS_3ffi5ShapeE10DLDataTypey"];Module["__ZNK3tvm7runtime6Tensor11CopyToBytesEPvm"]=wasmExports["_ZNK3tvm7runtime6Tensor11CopyToBytesEPvm"];Module["__ZN3tvm7runtime6Tensor13CopyFromBytesEPKvm"]=wasmExports["_ZN3tvm7runtime6Tensor13CopyFromBytesEPKvm"];Module["__ZNK3tvm7runtime6Tensor6CopyToERK8DLDeviceNS_3ffi8OptionalINS5_6StringEvEE"]=wasmExports["_ZNK3tvm7runtime6Tensor6CopyToERK8DLDeviceNS_3ffi8OptionalINS5_6StringEvEE"];Module["__ZN3tvm7runtime6Tensor10CopyFromToEPK8DLTensorPS2_Pv"]=wasmExports["_ZN3tvm7runtime6Tensor10CopyFromToEPK8DLTensorPS2_Pv"];Module["__ZN3tvm7runtime6Tensor15IsStorageSharedEPK8DLTensorS4_"]=wasmExports["_ZN3tvm7runtime6Tensor15IsStorageSharedEPK8DLTensorS4_"];Module["__ZN3tvm7runtime5Timer5StartE8DLDevice"]=wasmExports["_ZN3tvm7runtime5Timer5StartE8DLDevice"];Module["__ZN3tvm7runtime6detail14LogMessageImplERKNSt3__212basic_stringIcNS2_11char_traitsIcEENS2_9allocatorIcEEEEiiSA_"]=wasmExports["_ZN3tvm7runtime6detail14LogMessageImplERKNSt3__212basic_stringIcNS2_11char_traitsIcEENS2_9allocatorIcEEEEiiSA_"];Module["_TVMFFIDataTypeFromString"]=wasmExports["TVMFFIDataTypeFromString"];Module["_TVMFFIErrorSetRaisedFromCStr"]=wasmExports["TVMFFIErrorSetRaisedFromCStr"];Module["_TVMFFIErrorSetRaisedFromCStrParts"]=wasmExports["TVMFFIErrorSetRaisedFromCStrParts"];Module["_TVMFFIErrorSetRaised"]=wasmExports["TVMFFIErrorSetRaised"];Module["_TVMFFIErrorMoveFromRaised"]=wasmExports["TVMFFIErrorMoveFromRaised"];Module["_TVMFFIErrorCreate"]=wasmExports["TVMFFIErrorCreate"];Module["_TVMFFIErrorCreateWithCauseAndExtraContext"]=wasmExports["TVMFFIErrorCreateWithCauseAndExtraContext"];Module["_TVMFFIEnvCheckSignals"]=wasmExports["TVMFFIEnvCheckSignals"];Module["_TVMFFIEnvRegisterCAPI"]=wasmExports["TVMFFIEnvRegisterCAPI"];Module["_TVMFFIEnvSetDLPackManagedTensorAllocator"]=wasmExports["TVMFFIEnvSetDLPackManagedTensorAllocator"];Module["_TVMFFIEnvGetDLPackManagedTensorAllocator"]=wasmExports["TVMFFIEnvGetDLPackManagedTensorAllocator"];Module["_TVMFFIEnvTensorAlloc"]=wasmExports["TVMFFIEnvTensorAlloc"];Module["__ZN3tvm3ffi6Module19VisitContextSymbolsERKNS0_13TypedFunctionIFvNS0_6StringEPvEEE"]=wasmExports["_ZN3tvm3ffi6Module19VisitContextSymbolsERKNS0_13TypedFunctionIFvNS0_6StringEPvEEE"];Module["_TVMFFIEnvModRegisterContextSymbol"]=wasmExports["TVMFFIEnvModRegisterContextSymbol"];Module["_TVMFFIEnvModRegisterSystemLibSymbol"]=wasmExports["TVMFFIEnvModRegisterSystemLibSymbol"];Module["__ZN3tvm3ffi6Module12LoadFromFileERKNS0_6StringE"]=wasmExports["_ZN3tvm3ffi6Module12LoadFromFileERKNS0_6StringE"];Module["_TVMFFIFunctionCreate"]=wasmExports["TVMFFIFunctionCreate"];Module["_TVMFFIAnyViewToOwnedAny"]=wasmExports["TVMFFIAnyViewToOwnedAny"];Module["_TVMFFIFunctionSetGlobal"]=wasmExports["TVMFFIFunctionSetGlobal"];Module["_TVMFFIFunctionSetGlobalFromMethodInfo"]=wasmExports["TVMFFIFunctionSetGlobalFromMethodInfo"];Module["_TVMFFIFunctionCall"]=wasmExports["TVMFFIFunctionCall"];Module["_TVMFFIGetVersion"]=wasmExports["TVMFFIGetVersion"];Module["_TVMFFIObjectDecRef"]=wasmExports["TVMFFIObjectDecRef"];Module["_TVMFFIObjectIncRef"]=wasmExports["TVMFFIObjectIncRef"];Module["_TVMFFIObjectCreateOpaque"]=wasmExports["TVMFFIObjectCreateOpaque"];Module["_TVMFFITypeKeyToIndex"]=wasmExports["TVMFFITypeKeyToIndex"];Module["_TVMFFITypeRegisterField"]=wasmExports["TVMFFITypeRegisterField"];Module["_TVMFFITypeRegisterMethod"]=wasmExports["TVMFFITypeRegisterMethod"];Module["_TVMFFITypeRegisterMetadata"]=wasmExports["TVMFFITypeRegisterMetadata"];Module["_TVMFFITypeRegisterAttr"]=wasmExports["TVMFFITypeRegisterAttr"];Module["_TVMFFIGetTypeAttrColumn"]=wasmExports["TVMFFIGetTypeAttrColumn"];Module["_TVMFFIGetTypeInfo"]=wasmExports["TVMFFIGetTypeInfo"];Module["_TVMFFIStringFromByteArray"]=wasmExports["TVMFFIStringFromByteArray"];Module["_TVMFFIBytesFromByteArray"]=wasmExports["TVMFFIBytesFromByteArray"];Module["_TVMFFITensorCreateUnsafeView"]=wasmExports["TVMFFITensorCreateUnsafeView"];Module["_TVMFFITensorFromDLPack"]=wasmExports["TVMFFITensorFromDLPack"];Module["_TVMFFITensorFromDLPackVersioned"]=wasmExports["TVMFFITensorFromDLPackVersioned"];Module["_TVMFFITensorToDLPack"]=wasmExports["TVMFFITensorToDLPack"];Module["_TVMFFITensorToDLPackVersioned"]=wasmExports["TVMFFITensorToDLPackVersioned"];Module["___tvm_ffi_testing_dll_schema_id_int"]=wasmExports["__tvm_ffi_testing_dll_schema_id_int"];Module["___tvm_ffi_testing_dll_test_add_with_docstring"]=wasmExports["__tvm_ffi_testing_dll_test_add_with_docstring"];Module["_TVMFFITestingDummyTarget"]=wasmExports["TVMFFITestingDummyTarget"];Module["__ZN3tvm7runtime6memory7StorageC2ENS1_6BufferEPNS1_9AllocatorE"]=wasmExports["_ZN3tvm7runtime6memory7StorageC2ENS1_6BufferEPNS1_9AllocatorE"];Module["__ZN3tvm7runtime6memory10StorageObj17AllocTensorScopedExNS_3ffi5ShapeE10DLDataTypeNS3_6StringE"]=wasmExports["_ZN3tvm7runtime6memory10StorageObj17AllocTensorScopedExNS_3ffi5ShapeE10DLDataTypeNS3_6StringE"];Module["__ZN3tvm7runtime6memory10StorageObj11AllocTensorExNS_3ffi5ShapeE10DLDataType"]=wasmExports["_ZN3tvm7runtime6memory10StorageObj11AllocTensorExNS_3ffi5ShapeE10DLDataType"];Module["__ZN3tvm7runtime6memory13MemoryManager6GlobalEv"]=wasmExports["_ZN3tvm7runtime6memory13MemoryManager6GlobalEv"];Module["__ZN3tvm7runtime6memory13MemoryManager20GetOrCreateAllocatorE8DLDeviceNS1_13AllocatorTypeE"]=wasmExports["_ZN3tvm7runtime6memory13MemoryManager20GetOrCreateAllocatorE8DLDeviceNS1_13AllocatorTypeE"];Module["__ZN3tvm7runtime6memory13MemoryManager12GetAllocatorE8DLDeviceNS1_13AllocatorTypeE"]=wasmExports["_ZN3tvm7runtime6memory13MemoryManager12GetAllocatorE8DLDeviceNS1_13AllocatorTypeE"];Module["_TVMBackendAnyListSetPackedArg"]=wasmExports["TVMBackendAnyListSetPackedArg"];Module["_TVMBackendAnyListResetItem"]=wasmExports["TVMBackendAnyListResetItem"];Module["_TVMBackendAnyListMoveFromPackedReturn"]=wasmExports["TVMBackendAnyListMoveFromPackedReturn"];Module["__ZN3tvm7runtime2vm19TensorCacheMetadata11LoadFromStrERKNSt3__212basic_stringIcNS3_11char_traitsIcEENS3_9allocatorIcEEEESB_"]=wasmExports["_ZN3tvm7runtime2vm19TensorCacheMetadata11LoadFromStrERKNSt3__212basic_stringIcNS3_11char_traitsIcEENS3_9allocatorIcEEEESB_"];Module["__ZN3tvm7runtime2vm19TensorCacheMetadata4LoadERKNSt3__212basic_stringIcNS3_11char_traitsIcEENS3_9allocatorIcEEEE"]=wasmExports["_ZN3tvm7runtime2vm19TensorCacheMetadata4LoadERKNSt3__212basic_stringIcNS3_11char_traitsIcEENS3_9allocatorIcEEEE"];Module["__ZNK3tvm7runtime2vm19TensorCacheMetadata10FileRecord11ParamRecord4LoadE8DLDevicePKNSt3__212basic_stringIcNS6_11char_traitsIcEENS6_9allocatorIcEEEEPNS_3ffi8OptionalINS0_6TensorEvEE"]=wasmExports["_ZNK3tvm7runtime2vm19TensorCacheMetadata10FileRecord11ParamRecord4LoadE8DLDevicePKNSt3__212basic_stringIcNS6_11char_traitsIcEENS6_9allocatorIcEEEEPNS_3ffi8OptionalINS0_6TensorEvEE"];Module["__ZNK3tvm7runtime2vm19TensorCacheMetadata10FileRecord4LoadE8DLDeviceRKNSt3__212basic_stringIcNS5_11char_traitsIcEENS5_9allocatorIcEEEEPSB_PNS_3ffi8OptionalINS0_6TensorEvEE"]=wasmExports["_ZNK3tvm7runtime2vm19TensorCacheMetadata10FileRecord4LoadE8DLDeviceRKNSt3__212basic_stringIcNS5_11char_traitsIcEENS5_9allocatorIcEEEEPSB_PNS_3ffi8OptionalINS0_6TensorEvEE"];Module["_TVMBackendParallelLaunch"]=wasmExports["TVMBackendParallelLaunch"];Module["_TVMBackendParallelBarrier"]=wasmExports["TVMBackendParallelBarrier"];Module["__ZN3tvm7runtime6detail12LogFatalImplERKNSt3__212basic_stringIcNS2_11char_traitsIcEENS2_9allocatorIcEEEEiSA_"]=wasmExports["_ZN3tvm7runtime6detail12LogFatalImplERKNSt3__212basic_stringIcNS2_11char_traitsIcEENS2_9allocatorIcEEEEiSA_"];_free=wasmExports["free"];Module["__ZN3tvm7runtime6memory7StorageC1ENS1_6BufferEPNS1_9AllocatorE"]=wasmExports["_ZN3tvm7runtime6memory7StorageC1ENS1_6BufferEPNS1_9AllocatorE"];Module["_TVMWasmAllocSpace"]=wasmExports["TVMWasmAllocSpace"];Module["_TVMWasmFreeSpace"]=wasmExports["TVMWasmFreeSpace"];Module["_TVMFFIWasmFunctionCreate"]=wasmExports["TVMFFIWasmFunctionCreate"];Module["_TVMFFIWasmGetLastError"]=wasmExports["TVMFFIWasmGetLastError"];__initialize=Module["__initialize"]=wasmExports["_initialize"];_asyncify_start_unwind=wasmExports["asyncify_start_unwind"];_asyncify_stop_unwind=wasmExports["asyncify_stop_unwind"];_asyncify_start_rewind=wasmExports["asyncify_start_rewind"];_asyncify_stop_rewind=wasmExports["asyncify_stop_rewind"];wasmMemory=wasmExports["memory"];wasmExports["__indirect_function_table"];}var wasmImports={TVMFFIWasmFunctionDeleter:_TVMFFIWasmFunctionDeleter,TVMFFIWasmSafeCall:_TVMFFIWasmSafeCall,clock_time_get:_clock_time_get,emscripten_notify_memory_growth:_emscripten_notify_memory_growth,environ_get:_environ_get,environ_sizes_get:_environ_sizes_get,fd_close:_fd_close,fd_read:_fd_read,fd_seek:_fd_seek,fd_write:_fd_write};function callMain(args=[]){var entryFunction=__initialize;try{entryFunction();var ret=0;exitJS(ret,true);return ret}catch(e){return handleException(e)}}function run(args=arguments_){if(runDependencies>0){dependenciesFulfilled=run;return}preRun();if(runDependencies>0){dependenciesFulfilled=run;return}function doRun(){Module["calledRun"]=true;if(ABORT)return;initRuntime();Module["onRuntimeInitialized"]?.();var noInitialRun=Module["noInitialRun"]||false;if(!noInitialRun)callMain(args);postRun();}if(Module["setStatus"]){Module["setStatus"]("Running...");setTimeout(()=>{setTimeout(()=>Module["setStatus"](""),1);doRun();},1);}else {doRun();}}var wasmExports;createWasm();run();

        this.Module = Module;
        this.start = Module.wasmLibraryProvider.start;
        this.imports = Module.wasmLibraryProvider.imports;
        this.wasiImport = this.imports["wasi_snapshot_preview1"];
    }

    /**
     * Get performance measurement.
     */
    function getPerformance() {
        if (typeof performance === "undefined") {
            const performanceNode = require("perf_hooks");
            return performanceNode.performance;
        }
        else {
            return performance;
        }
    }
    /**
     * Create a new websocket for a given URL
     * @param url The url.
     */
    function createWebSocket(url) {
        if (typeof WebSocket === "undefined") {
            const WebSocket = require("ws");
            return new WebSocket(url);
        }
        else {
            return new WebSocket(url);
        }
    }
    /**
     * Create a WASI based on current environment.
     *
     * @return A wasi that can run on broswer or local.
     */
    function createPolyfillWASI() {
        return new EmccWASI();
    }

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
    /**
     * @internal
     * FFI Library wrapper, maintains most runtime states.
     */
    class FFILibrary {
        constructor(wasmInstance, imports) {
            this.recycledCallStacks = [];
            this.wasmInstance = wasmInstance;
            this.memory = new Memory(this.detectWasmMemory(this.wasmInstance, imports));
            assert(this.wasmInstance.exports !== undefined, "Expect the library module contains exports");
            this.exports = this.wasmInstance.exports;
            this.wasm32 = this.memory.wasm32;
            this.validateInstance();
        }
        dispose() {
            var _a;
            while (this.recycledCallStacks.length != 0) {
                this.recycledCallStacks.pop().dispose();
            }
            (_a = this.webGPUContext) === null || _a === void 0 ? void 0 : _a.dispose();
        }
        sizeofPtr() {
            return this.memory.sizeofPtr();
        }
        checkCall(code) {
            if (code != 0) {
                const msgPtr = this.exports
                    .TVMFFIWasmGetLastError();
                throw new Error(this.memory.loadCString(msgPtr));
            }
        }
        getOrAllocCallStack() {
            if (this.recycledCallStacks.length != 0) {
                return this.recycledCallStacks.pop();
            }
            return new CachedCallStack(this.memory, this.exports.TVMWasmAllocSpace, this.exports.TVMWasmFreeSpace);
        }
        recycleCallStack(callstack) {
            callstack.reset();
            this.recycledCallStacks.push(callstack);
        }
        validateInstance() {
            this.checkExports(["TVMWasmAllocSpace", "TVMWasmFreeSpace"]);
        }
        checkExports(funcNames) {
            const missList = [];
            for (const name of funcNames) {
                const f = this.exports[name];
                if (!(f instanceof Function)) {
                    missList.push(name);
                }
            }
            if (missList.length != 0) {
                throw new Error("Cannot find " + missList + " in exports");
            }
        }
        detectWasmMemory(instance, imports) {
            if (instance.exports.memory instanceof WebAssembly.Memory) {
                return instance.exports.memory;
            }
            if (imports.env && imports.env.memory instanceof WebAssembly.Memory) {
                return imports.env.memory;
            }
            throw new Error("Cannt detect wasm memory from imports " +
                imports +
                " or exports" +
                instance.exports);
        }
    }
    /**
     * @internal
     * Manages extra runtime context for the runtime.
     */
    class RuntimeContext {
        constructor(getGlobalFunc) {
            this.autoDisposeScope = [];
            this.functionListGlobalNamesFunctor = getGlobalFunc("ffi.FunctionListGlobalNamesFunctor");
            this.moduleGetFunction = getGlobalFunc("ffi.ModuleGetFunction");
            this.moduleImport = getGlobalFunc("ffi.ModuleImportModule");
            this.tensorEmpty = getGlobalFunc("runtime.TVMTensorAllocWithScope");
            this.tensorCopyFromTo = getGlobalFunc("runtime.TVMTensorCopyFromTo");
            this.tensorCopyFromJSBytes = getGlobalFunc("tvmjs.runtime.TensorCopyFromBytes");
            this.tensorCopyToJSBytes = getGlobalFunc("tvmjs.runtime.TensorCopyToBytes");
            this.arrayGetItem = getGlobalFunc("ffi.ArrayGetItem");
            this.arrayGetSize = getGlobalFunc("ffi.ArraySize");
            this.arrayMake = getGlobalFunc("ffi.Array");
            this.arrayConcat = getGlobalFunc("tvmjs.runtime.ArrayConcat");
            this.getSysLib = getGlobalFunc("ffi.SystemLib");
            this.tensorCacheGet = getGlobalFunc("vm.builtin.tensor_cache.get");
            this.tensorCacheRemove = getGlobalFunc("vm.builtin.tensor_cache.remove");
            this.tensorCacheUpdate = getGlobalFunc("vm.builtin.tensor_cache.update");
            this.tensorCacheClear = getGlobalFunc("vm.builtin.tensor_cache.clear");
            this.arrayDecodeStorage = getGlobalFunc("tvmjs.array.decode_storage");
            this.paramModuleFromCache = getGlobalFunc("vm.builtin.param_module_from_cache");
            this.paramModuleFromCacheByName = getGlobalFunc("vm.builtin.param_module_from_cache_by_name");
            this.makeShapeTuple = getGlobalFunc("ffi.Shape");
            this.tensorCreateView = getGlobalFunc("runtime.TVMTensorCreateView");
            this.sampleTopPFromLogits = getGlobalFunc("vm.builtin.sample_top_p_from_logits");
            this.sampleTopPFromProb = getGlobalFunc("vm.builtin.sample_top_p_from_prob");
            this.applyRepetitionPenalty = getGlobalFunc("vm.builtin.apply_repetition_penalty");
            this.applyPresenceAndFrequencyPenalty = getGlobalFunc("vm.builtin.apply_presence_and_frequency_penalty");
            this.applySoftmaxWithTemperature = getGlobalFunc("vm.builtin.apply_softmax_with_temperature");
            this.concatEmbeddings = getGlobalFunc("tvmjs.runtime.ConcatEmbeddings");
        }
        dispose() {
            var _a;
            // call array cache clear to clear all cached items
            this.tensorCacheClear.dispose();
            this.arrayGetItem.dispose();
            this.arrayGetSize.dispose();
            this.arrayMake.dispose();
            this.arrayConcat.dispose();
            this.tensorCacheGet.dispose();
            this.tensorCacheRemove.dispose();
            this.tensorCacheUpdate.dispose();
            this.arrayDecodeStorage.dispose();
            this.paramModuleFromCache.dispose();
            this.paramModuleFromCacheByName.dispose();
            this.makeShapeTuple.dispose();
            this.tensorCreateView.dispose();
            this.sampleTopPFromLogits.dispose();
            this.applyRepetitionPenalty.dispose();
            this.applyPresenceAndFrequencyPenalty.dispose();
            this.applySoftmaxWithTemperature.dispose();
            (_a = this.concatEmbeddings) === null || _a === void 0 ? void 0 : _a.dispose();
        }
        beginScope() {
            this.autoDisposeScope.push([]);
        }
        endScope() {
            if (this.autoDisposeScope.length === 0) {
                throw Error("tvm.endScope called when the stack is empty.");
            }
            // automatically dispose all the tracked values in the current scope.
            const currScope = this.autoDisposeScope.pop();
            for (let i = 0; i < currScope.length; ++i) {
                const val = currScope[i];
                if (val !== undefined) {
                    val.dispose();
                }
            }
        }
        /**
         * Track object for dispose in current scope.
         *
         * @param obj The object to be tracked.
         * @returns the same object.
          * Note: This function only needs to be called for raw system C API values.
         *       The return value of PackedFunc will be automatically tracked.
         */
        attachToCurrentScope(obj) {
            if (this.autoDisposeScope.length === 0) {
                throw Error("Must call beginScope to use functions that returns TVM objects");
            }
            const currScope = this.autoDisposeScope[this.autoDisposeScope.length - 1];
            currScope.push(obj);
            return obj;
        }
        moveToParentScope(obj) {
            this.detachFromCurrentScope(obj);
            if (this.autoDisposeScope.length < 2) {
                throw Error("moveToParentScope: Parent scope do not exist");
            }
            const parentScope = this.autoDisposeScope[this.autoDisposeScope.length - 2];
            parentScope.push(obj);
            return obj;
        }
        detachFromCurrentScope(obj) {
            const currScope = this.autoDisposeScope[this.autoDisposeScope.length - 1];
            let occurrence = 0;
            for (let i = 0; i < currScope.length; ++i) {
                if (currScope[i] === obj) {
                    occurrence += 1;
                    currScope[i] = undefined;
                }
            }
            if (occurrence === 0) {
                throw Error("Cannot find obj in the current auto conversion pool");
            }
            if (occurrence > 1) {
                throw Error("Value attached to scope multiple times");
            }
            return obj;
        }
    }
    /**
     * A typed scalar constant used to represent a typed number
     * argument to PackedFunc calls.
     */
    class Scalar {
        constructor(value, dtype) {
            this.value = value;
            this.dtype = dtype;
        }
    }
    const DeviceEnumToStr = {
        1: "cpu",
        2: "cuda",
        4: "opencl",
        8: "metal",
        15: "webgpu"
    };
    const DeviceStrToEnum = {
        cpu: 1,
        cuda: 2,
        cl: 4,
        opencl: 4,
        vulkan: 7,
        metal: 8,
        webgpu: 15
    };
    /**
     * Represent a runtime context where a Tensor can reside.
     */
    class DLDevice {
        constructor(deviceType, deviceId, lib) {
            const tp = typeof deviceType;
            if (tp === "string") {
                this.deviceType = DeviceStrToEnum[deviceType];
                if (this.deviceType === undefined) {
                    throw new Error("Cannot recogonize deviceType " + deviceType);
                }
            }
            else if (tp === "number") {
                this.deviceType = deviceType;
            }
            else {
                throw new Error("Cannot take type " + tp + " as deviceType");
            }
            this.deviceId = deviceId;
            this.lib = lib;
        }
        /**
         * Synchronize the device
         */
        sync() {
            return __awaiter(this, void 0, void 0, function* () {
                if (this.deviceType === DeviceStrToEnum.webgpu) {
                    assert(this.lib.webGPUContext !== undefined);
                    yield this.lib.webGPUContext.sync();
                }
            });
        }
        toString() {
            return (DeviceEnumToStr[this.deviceType] + ":" + this.deviceId.toString());
        }
    }
    /**
     * The data type code in DLDataType
     */
    var DLDataTypeCode;
    (function (DLDataTypeCode) {
        DLDataTypeCode[DLDataTypeCode["Int"] = 0] = "Int";
        DLDataTypeCode[DLDataTypeCode["UInt"] = 1] = "UInt";
        DLDataTypeCode[DLDataTypeCode["Float"] = 2] = "Float";
        DLDataTypeCode[DLDataTypeCode["OpaqueHandle"] = 3] = "OpaqueHandle";
    })(DLDataTypeCode || (DLDataTypeCode = {}));
    const DLDataTypeCodeToStr = {
        0: "int",
        1: "uint",
        2: "float",
        3: "handle",
    };
    /**
     * Runtime data type of Tensor.
     */
    class DLDataType {
        constructor(code, bits, lanes) {
            this.code = code;
            this.bits = bits;
            this.lanes = lanes;
        }
        toString() {
            const ret = DLDataTypeCodeToStr[this.code] + this.bits.toString();
            if (this.lanes != 1) {
                return ret + "x" + this.lanes.toString();
            }
            else {
                return ret;
            }
        }
        numStorageBytes() {
            return (this.bits * this.lanes + 7) >> 3;
        }
    }
    /**
     * Generic object base
     */
    class TVMObject {
        constructor(handle, lib, ctx) {
            this.handle = handle;
            this.lib = lib;
            this.ctx = ctx;
        }
        dispose() {
            if (this.handle != 0) {
                this.lib.checkCall(this.lib.exports.TVMFFIObjectDecRef(this.handle));
                this.handle = 0;
            }
        }
        /**
         * Get handle of module, check it is not null.
         *
         * @param requireNotNull require handle is not null.
         * @returns The handle.
         */
        getHandle(requireNotNull = true) {
            if (requireNotNull && this.handle === 0) {
                throw Error("Object has already been disposed");
            }
            return this.handle;
        }
        /** get the type index of the object */
        typeIndex() {
            if (this.handle === 0) {
                throw Error("The current Object has already been disposed");
            }
            return this.lib.memory.loadObjectTypeIndex(this.handle);
        }
        /** get the type key of the object */
        typeKey() {
            const type_index = this.typeIndex();
            const typeInfoPtr = this.lib.exports.TVMFFIGetTypeInfo(type_index);
            return this.lib.memory.loadTypeInfoTypeKey(typeInfoPtr);
        }
    }
    /**
     * Cell holds the PackedFunc object.
     */
    class PackedFuncCell extends TVMObject {
        constructor(handle, lib, ctx) {
            super(handle, lib, ctx);
        }
    }
    /**
     * Tensor( n-dimnesional array).
     */
    class Tensor extends TVMObject {
        constructor(handle, lib, ctx, isView) {
            // if the array is a view, we need to create a new object with a null handle
            // so dispose won't trigger memory free
            const objectHandle = isView ? 0 : handle;
            super(objectHandle, lib, ctx);
            this.isView = isView;
            if (this.isView) {
                this.dltensor = handle;
            }
            else {
                this.dltensor = this.getDLTensorFromArrayHandle(this.handle);
            }
            // constant offsets.
            const arrayOffsetData = 0;
            const arrayOffsetContext = arrayOffsetData + this.lib.sizeofPtr();
            const arrayOffsetDevType = arrayOffsetContext;
            const arrayOffsetDevId = arrayOffsetContext + 4 /* SizeOf.I32 */;
            const arrayOffsetNdim = arrayOffsetContext + 8 /* SizeOf.DLDevice */;
            const arrayOffsetDtype = arrayOffsetNdim + 4 /* SizeOf.I32 */;
            const arrayOffsetDtypeCode = arrayOffsetDtype;
            const arrayOffsetDtypeBits = arrayOffsetDtype + 1 /* SizeOf.U8 */;
            const arrayOffsetDtypeLanes = arrayOffsetDtypeBits + 1 /* SizeOf.U8 */;
            const arrayOffsetShape = arrayOffsetDtype + 4 /* SizeOf.DLDataType */;
            const arrayOffsetStrides = arrayOffsetShape + this.lib.sizeofPtr();
            const arrayOffsetByteOffset = arrayOffsetStrides + this.lib.sizeofPtr();
            // dataPtr
            this.dataPtr = lib.memory.loadPointer(this.dltensor);
            // ndim
            this.ndim = lib.memory.loadI32(this.dltensor + arrayOffsetNdim);
            // shape
            const cshapePtr = lib.memory.loadPointer(this.dltensor + arrayOffsetShape);
            this.shape = [];
            for (let i = 0; i < this.ndim; ++i) {
                this.shape.push(lib.memory.loadI64(cshapePtr + i * 8 /* SizeOf.I64 */));
            }
            // dtype
            const code = lib.memory.loadU8(this.dltensor + arrayOffsetDtypeCode);
            const bits = lib.memory.loadU8(this.dltensor + arrayOffsetDtypeBits);
            const lanes = lib.memory.loadU16(this.dltensor + arrayOffsetDtypeLanes);
            this.dlDataType = new DLDataType(code, bits, lanes);
            this.dtype = this.dlDataType.toString();
            // device
            const deviceType = lib.memory.loadI32(this.dltensor + arrayOffsetDevType);
            const deviceId = lib.memory.loadI32(this.dltensor + arrayOffsetDevId);
            this.device = new DLDevice(deviceType, deviceId, lib);
            // byte_offset
            this.byteOffset = lib.memory.loadI64(this.dltensor + arrayOffsetByteOffset);
        }
        /**
         * Create a view of the array.
         * @param shape The shape of the view.
         * @param dtype The data type of the new array.
         * @returns The new sliced ndarray.
         */
        view(shape, dtype) {
            const shapeArray = shape.map((value) => new Scalar(value, "int"));
            if (dtype === undefined) {
                dtype = this.dtype;
            }
            return this.ctx.tensorCreateView(this, this.ctx.makeShapeTuple(...shapeArray), this.dtype, 
            /*relative_byte_offset=*/ new Scalar(0, "int"));
        }
        /**
         * Get dataPtr of NDarray
         *
         * @returns The handle.
         */
        getDataPtr() {
            if (this.handle === 0) {
                throw Error("Tensor has already been disposed");
            }
            return this.dataPtr;
        }
        /**
         * Copy data from another Tensor or javascript array.
         * The number of elements must match.
         *
         * @param data The source data array.
         * @returns this
         */
        copyFrom(data) {
            if (data instanceof Tensor) {
                this.ctx.tensorCopyFromTo(data, this);
                return this;
            }
            else {
                const size = this.shape.reduce((a, b) => {
                    return a * b;
                }, 1);
                if (data.length != size) {
                    throw new Error("data size and shape mismatch data.length" +
                        data.length +
                        " vs " +
                        size);
                }
                let buffer;
                if (this.dtype === "float32") {
                    buffer = Float32Array.from(data).buffer;
                }
                else if (this.dtype === "float64") {
                    buffer = Float64Array.from(data).buffer;
                }
                else if (this.dtype === "int32") {
                    buffer = Int32Array.from(data).buffer;
                }
                else if (this.dtype === "int8") {
                    buffer = Int8Array.from(data).buffer;
                }
                else if (this.dtype === "uint8") {
                    buffer = Uint8Array.from(data).buffer;
                }
                else if (this.dtype === "uint32") {
                    buffer = Uint32Array.from(data).buffer;
                }
                else {
                    throw new Error("Unsupported data type " + this.dtype);
                }
                return this.copyFromRawBytes(new Uint8Array(buffer));
            }
        }
        /**
         * Copy data from raw bytes.
         * @param data Uint8Array of bytes.
         * @returns this
         */
        copyFromRawBytes(data) {
            var _a;
            // short cut for gpu copy
            if (this.device.deviceType === DeviceStrToEnum.webgpu) {
                (_a = this.lib.webGPUContext) === null || _a === void 0 ? void 0 : _a.copyRawBytesToBuffer(data, this.getDataPtr(), 0, data.length);
                return this;
            }
            // CPU copy
            const size = this.shape.reduce((a, b) => {
                return a * b;
            }, 1);
            const nbytes = this.dlDataType.numStorageBytes() * size;
            if (nbytes != data.length) {
                throw new Error("Expect the data's length equals nbytes=" + nbytes);
            }
            this.ctx.tensorCopyFromJSBytes(this, data);
            return this;
        }
        /**
         * Return a copied Uint8Array of the raw bytes in the Tensor.
         * @returns The result array.
         */
        toRawBytes() {
            if (this.device.deviceType != DeviceStrToEnum.cpu) {
                throw new Error("Can only sync copy CPU array, use cpu_arr.copyfrom(gpu_arr) then sync instead.");
            }
            return this.ctx.tensorCopyToJSBytes(this);
        }
        /**
         * Return a TypedArray copy of the Tensor, the specific type depends on
         * the dtype of the Tensor.
         * @returns The result array.
         */
        toArray() {
            const stype = this.dtype;
            if (stype === "float32") {
                return new Float32Array(this.toRawBytes().buffer);
            }
            else if (stype === "float64") {
                return new Float64Array(this.toRawBytes().buffer);
            }
            else if (stype === "int32") {
                return new Int32Array(this.toRawBytes().buffer);
            }
            else if (stype === "int8") {
                return new Int8Array(this.toRawBytes().buffer);
            }
            else if (stype === "uint8") {
                return new Uint8Array(this.toRawBytes().buffer);
            }
            else {
                throw new Error("Unsupported data type " + this.dtype);
            }
        }
        getDLTensorFromArrayHandle(handle) {
            return handle + 24 /* SizeOf.ObjectHeader */;
        }
    }
    /**
     * Runtime Module.
     */
    class Module extends TVMObject {
        constructor(handle, lib, ctx) {
            super(handle, lib, ctx);
        }
        /**
         * Get a function in the module.
         * @param name The name of the function.
         * @param queryImports Whether to also query imports
         * @returns The result function.
         */
        getFunction(name, queryImports = true) {
            return this.ctx.moduleGetFunction(this, name, queryImports);
        }
        /**
         * Import another module into the current runtime module.
         * @param mod The module to be imported.
         */
        importModule(mod) {
            this.ctx.moduleImport(this, mod);
        }
    }
    /** Runtime array object. */
    class TVMArray extends TVMObject {
        constructor(handle, lib, ctx) {
            super(handle, lib, ctx);
        }
        /**
         * @returns the size of the array.
         */
        size() {
            return this.ctx.arrayGetSize(this);
        }
        /**
         * Get index-th element of the array
         * @param index the array index.
         * @returns The element.
         */
        get(index) {
            return this.ctx.arrayGetItem(this, new Scalar(index, "int32"));
        }
    }
    var VMAllocatorKind;
    (function (VMAllocatorKind) {
        VMAllocatorKind[VMAllocatorKind["NAIVE_ALLOCATOR"] = 1] = "NAIVE_ALLOCATOR";
        VMAllocatorKind[VMAllocatorKind["POOLED_ALLOCATOR"] = 2] = "POOLED_ALLOCATOR";
    })(VMAllocatorKind || (VMAllocatorKind = {}));
    /**
     *  VirtualMachine Executor.
     *
     *  This is a thin wrapper of the underlying TVM module.
     *  you can also directly call set_input, run, and get_output
     *  of underlying module functions
     */
    class VirtualMachine {
        /**
         * Constructor
         * @param mod The underlying module, need to be detached.
         * @param device The main device ro run VM on.
         */
        constructor(mod, device) {
            this.mod = mod;
            this.mod.getFunction("vm_initialization")(new Scalar(device.deviceType, "int"), new Scalar(device.deviceId, "int"), new Scalar(VMAllocatorKind.POOLED_ALLOCATOR, "int"), 
            // explicitly specify host device type
            new Scalar(DeviceStrToEnum.cpu, "int"), new Scalar(0, "int"), new Scalar(VMAllocatorKind.POOLED_ALLOCATOR, "int"));
        }
        dispose() {
            this.mod.dispose();
        }
        /**
         * Get a function in the VM module.
         * @param name The name of the function.
         * @returns The result function.
         */
        getFunction(name) {
            return this.mod.getFunction(name);
        }
        /**
         * Get the internal module.
         */
        getInternalModule() {
            return this.mod;
        }
    }
    /** Code used as the first argument of the async callback. */
    var AsyncCallbackCode;
    (function (AsyncCallbackCode) {
        AsyncCallbackCode[AsyncCallbackCode["kReturn"] = 4] = "kReturn";
        AsyncCallbackCode[AsyncCallbackCode["kException"] = 5] = "kException";
    })(AsyncCallbackCode || (AsyncCallbackCode = {}));
    /**
     * TVM runtime instance.
     *
     * All objects(Tensor, Module, PackedFunc) returned by TVM runtim function call
     * and PackedFunc instance are tracked through a scope mechanism that will get
     * auto-released when we call EndScope.
     *
     * This is necessarily to be able to release the underlying WASM and WebGPU memory that
     * are not tracked through JS native garbage collection mechanism.
     *
     * This does mean that we have to get familar with the following functions:
     * - {@link beginScope}
     * - {@link endScope}
     * - {@link withNewScope}
     * - {@link attachToCurrentScope}
     * - {@link detachFromCurrentScope}
     */
    class Instance {
        /**
         * Constructor
         *
         * importObject can also be a {@link LibraryProvider} object,
         * a WASI object, or an object containing wasmLibraryProvider field.
         *
         * @param wasmModule The input module or instance.
         * @param importObject The imports to initialize the wasmInstance if it is not provided.
         * @param wasmInstance Additional wasm instance argument for deferred construction.
         * @param env Directly specified environment module.
         *
         * @see Please use the async version {@link instantiate} when targeting browsers.
         */
        constructor(wasmModule, importObject = {}, wasmInstance, env) {
            this.cacheMetadata = {};
            this.initProgressCallback = [];
            this.deviceLostIsError = true; // whether device.lost is due to actual error or dispose()
            this.cacheState = new CacheState();
            if (wasmInstance instanceof WebAssembly.Instance) {
                assert(env instanceof Environment, "env must be provided when passing in instance");
            }
            else {
                assert(env === undefined);
                env = new Environment(importObject);
                wasmInstance = new WebAssembly.Instance(wasmModule, env.imports);
            }
            env.start(wasmInstance);
            this.env = env;
            this.lib = new FFILibrary(wasmInstance, env.imports);
            this.memory = this.lib.memory;
            this.exports = this.lib.exports;
            this.asyncifyHandler = new AsyncifyHandler(this.exports, this.memory.memory);
            this.objFactory = new Map();
            this.ctx = new RuntimeContext((name) => {
                const autoAttachToScope = false;
                // runtime context function do not auto-release.
                return this.getGlobalFuncInternal(name, autoAttachToScope);
            });
            this.registerEnvGlobalPackedFuncs();
            this.registerObjectFactoryFuncs();
            this.rng = new LinearCongruentialGenerator();
        }
        /**
         * Benchmark stable execution of the run function.
         *
          * @param run The run function.
          * @param dev The device to sync during each run.
          * @param number The number of times to compute the average.
          * @param repeat The number of times to repeat the run.
         */
        benchmark(run_1, dev_1) {
            return __awaiter(this, arguments, void 0, function* (run, dev, number = 10, repeat = 1) {
                // Skip first run as it can involve GPU warmup and module loading time.
                const perf = getPerformance();
                const results = [];
                // run with new scope
                this.withNewScope(run);
                yield dev.sync();
                for (let k = 0; k < repeat; ++k) {
                    const tstart = perf.now();
                    for (let i = 0; i < number; ++i) {
                        this.withNewScope(run);
                    }
                    yield dev.sync();
                    const tend = perf.now();
                    results.push((tend - tstart) / number);
                }
                return results;
            });
        }
        /**
         * Check whether we enabled asyncify mode
         * @returns The asynctify mode toggle
         */
        asyncifyEnabled() {
            return this.asyncifyHandler.enabled();
        }
        dispose() {
            this.deviceLostIsError = false; // prevent dispose to trigger device.lost error
            // order matters
            // dispose caches before ctx
            this.cacheState.dispose();
            // ctx release goes back into lib.
            this.ctx.dispose();
            this.lib.dispose();
            // Cannot set deviceLostIsError back to true here because GPUDevice.destroy() is asynchronous.
        }
        /**
         * Obtain the runtime information in readable format.
         */
        runtimeStatsText() {
            if (this.lib.webGPUContext !== undefined) {
                return this.lib.webGPUContext.runtimeStatsText();
            }
            else {
                return "";
            }
        }
        /**
         * Begin a new scope for tracking object disposal.
         */
        beginScope() {
            this.ctx.beginScope();
        }
        /**
         * End a scope and release all created TVM objects
         * under the current scope.
         *
         * Exception: one can call {@link moveToParentScope} to move
         * a value to parent scope.
         */
        endScope() {
            this.ctx.endScope();
        }
        /**
         * Perform action under a new scope.
         *
         * @param action The action function.
         * @returns The result value.
         *
          * Note: For action to return a valid value,
         *       we will need to call {@link moveToParentScope}
         *       for the objects that are created in the scope.
         */
        withNewScope(action) {
            this.beginScope();
            const val = action();
            this.endScope();
            return val;
        }
        /**
         * Attach a detached obj to the auto-release pool of the current scope.
         *
         * @param obj The input obj.
          * Note: Normally user do not need to call this function explicitly, as
         *       all library call return values are explicitly attached to
         *       the current scope. You only need to do so when you call
         *       {@link detachFromCurrentScope} to create a detached object.
         */
        attachToCurrentScope(obj) {
            return this.ctx.attachToCurrentScope(obj);
        }
        /**
         * Move obj's attachment to the parent scope.
         *
         * This function is useful to make sure objects are still
         * alive when exit the current scope.
         *
         * @param obj The object to be moved.
         * @returns The input obj.
         */
        moveToParentScope(obj) {
            return this.ctx.moveToParentScope(obj);
        }
        /**
         * Detach the object from the current scope
         * so it won't be released via auto-release during endscope.
         *
         * User needs to either explicitly call obj.dispose(), or
         * {@link attachToCurrentScope} to re-attach to the current scope.
         *
         * This function can be used to return values to the parent scope.
         * @param obj The object.
         */
        detachFromCurrentScope(obj) {
            return this.ctx.detachFromCurrentScope(obj);
        }
        /**
         * Get system-wide library module in the wasm.
         * System lib is a global module that contains self register functions in startup.
         * @returns The system library module.
         */
        systemLib() {
            return this.ctx.getSysLib();
        }
        /**
         * List all the global function names registered in the runtime.
         * @returns The name list.
         */
        listGlobalFuncNames() {
            return this.withNewScope(() => {
                const functor = this.ctx.functionListGlobalNamesFunctor();
                const numNames = functor(new Scalar(-1, "int"));
                const names = new Array(numNames);
                for (let i = 0; i < numNames; i++) {
                    names[i] = functor(new Scalar(i, "int"));
                }
                return names;
            });
        }
        /**
         * Register function to be global function in tvm runtime.
         * @param name The name of the function.
          * @param func Function to be registered.
         * @param override Whether overwrite function in existing registry.
         */
        registerFunc(name, func, override = false) {
            this.withNewScope(() => {
                const autoAttachToScope = true;
                // packed func can be released once it is registered
                const packedFunc = this.toPackedFuncInternal(func, autoAttachToScope);
                const ioverride = override ? 1 : 0;
                const stack = this.lib.getOrAllocCallStack();
                const nameOffset = stack.allocByteArrayForString(name);
                stack.commitToWasmMemory();
                this.lib.checkCall(this.lib.exports.TVMFFIFunctionSetGlobal(stack.ptrFromOffset(nameOffset), packedFunc._tvmPackedCell.getHandle(), ioverride));
                this.lib.recycleCallStack(stack);
            });
        }
        /**
         * Get global PackedFunc from the runtime.
         * @param name The name of the function.
         * @returns The result function.
         */
        getGlobalFunc(name) {
            return this.getGlobalFuncInternal(name, true);
        }
        getGlobalFuncInternal(name, autoAttachToScope = true) {
            const stack = this.lib.getOrAllocCallStack();
            const nameOffset = stack.allocByteArrayForString(name);
            const outOffset = stack.allocPtrArray(1);
            const outPtr = stack.ptrFromOffset(outOffset);
            stack.commitToWasmMemory(outOffset);
            this.lib.checkCall(this.exports.TVMFFIFunctionGetGlobal(stack.ptrFromOffset(nameOffset), outPtr));
            const handle = this.memory.loadPointer(outPtr);
            this.lib.recycleCallStack(stack);
            if (handle === 0) {
                throw Error("Cannot find global function " + name);
            }
            const ret = this.makePackedFunc(handle);
            if (autoAttachToScope)
                this.ctx.attachToCurrentScope(ret);
            return ret;
        }
        /**
         * Check if func is PackedFunc.
         *
         * @param func The input.
         * @returns The check result.
         */
        isPackedFunc(func) {
            return typeof func === "function" && func.hasOwnProperty("_tvmPackedCell");
        }
        /**
         * Convert func to PackedFunc
         *
         * @param func Input function.
         * @returns The converted function.
         */
        toPackedFunc(func) {
            return this.toPackedFuncInternal(func, true);
        }
        toPackedFuncInternal(func, autoAttachToScope) {
            if (this.isPackedFunc(func))
                return func;
            const ret = this.createPackedFuncFromSafeCallType(this.wrapJSFuncAsSafeCallType(func));
            if (autoAttachToScope)
                return this.ctx.attachToCurrentScope(ret);
            return ret;
        }
        /**
        * Setup a virtual machine module with given device.
        *
        * @param dev DLDevice the device.
        * @returns The created virtual machime.
        */
        createVirtualMachine(dev) {
            const mod = this.ctx.detachFromCurrentScope(this.systemLib().getFunction("vm_load_executable")());
            return this.ctx.attachToCurrentScope(new VirtualMachine(mod, dev));
        }
        //-----------------------------------------------
        // Native Tensor Cache Support
        //-----------------------------------------------
        /**
         * Register a call back for fetch progress.
        *
         * @param cb the fetch progress callback.
         */
        registerInitProgressCallback(cb) {
            this.initProgressCallback.push(cb);
        }
        /**
         * Get parameters in the form of prefix_i
         *
         * @param prefix The parameter prefix.
         * @param numParams  Number of parameters.
         * @returns
         */
        getParamsFromCache(prefix, numParams) {
            return this.ctx.paramModuleFromCache(prefix, new Scalar(numParams, "int32")).getFunction("get_params")();
        }
        /**
         * Get parameters based on parameter names provided
         *
         * @param paramNames Names of the parameters.
         * @returns Parameters read.
         */
        getParamsFromCacheByName(paramNames) {
            return this.ctx.paramModuleFromCacheByName(paramNames).getFunction("get_params")();
        }
        /**
         * Get Tensor from cache.
         * @param name  The name of array.
         * @returns  The result.
         */
        tensorCacheGet(name) {
            return this.ctx.tensorCacheGet(name);
        }
        /**
         * Get Tensor from cache.
         * @param name  The name of array.
         * @returns  The result.
         */
        tensorCacheRemove(name) {
            return this.ctx.tensorCacheRemove(name);
        }
        /**
         * Update the tensor cache.
         * @param name The name of the array.
         * @param arr The content.
         */
        tensorCacheUpdate(name, arr, override = false) {
            this.ctx.tensorCacheUpdate(name, arr, this.scalar(override ? 1 : 0, "int32"));
        }
        /**
         * Clear the tensor cache.
         */
        tensorCacheClear() {
            this.ctx.tensorCacheClear();
        }
        fetchTensorCache(tensorCacheUrl_1, device_1) {
            return __awaiter(this, arguments, void 0, function* (tensorCacheUrl, device, cacheScopeOrOptions = "tvmjs", cacheType = "cache", signal) {
                var _a, _b, _c;
                let options;
                if (typeof cacheScopeOrOptions === "object" && cacheScopeOrOptions !== null) {
                    options = cacheScopeOrOptions;
                }
                else {
                    options = {
                        cacheScope: cacheScopeOrOptions,
                        signal: signal,
                    };
                }
                const cacheScope = (_a = options.cacheScope) !== null && _a !== void 0 ? _a : "tvmjs";
                const artifactCache = createArtifactCache(cacheScope, Object.assign(Object.assign({}, options), { cacheType: (_b = options.cacheType) !== null && _b !== void 0 ? _b : cacheType }));
                const effectiveSignal = (_c = options.signal) !== null && _c !== void 0 ? _c : signal;
                const jsonUrl = new URL("tensor-cache.json", tensorCacheUrl).href;
                const list = yield artifactCache.fetchWithCache(jsonUrl, "json", effectiveSignal);
                yield this.fetchTensorCacheInternal(tensorCacheUrl, list["records"], device, artifactCache, effectiveSignal);
                this.cacheMetadata = Object.assign(Object.assign({}, this.cacheMetadata), list["metadata"]);
            });
        }
        /**
         * Fetch list of Tensor into the TensorCache.
         *
         * @param tensorCacheUrl The cache url.
         * @param list The list of array data.
         * @param device The device to store the data to.
         * @param artifactCache The artifact cache
         * @param signal An optional AbortSignal to abort the fetch
         */
        fetchTensorCacheInternal(tensorCacheUrl, list, device, artifactCache, signal) {
            return __awaiter(this, void 0, void 0, function* () {
                const perf = getPerformance();
                const tstart = perf.now();
                let totalBytes = 0;
                for (let i = 0; i < list.length; ++i) {
                    totalBytes += list[i].nbytes;
                }
                let fetchedBytes = 0;
                let fetchedShards = 0;
                let timeElapsed = 0;
                const cacheOnly = yield artifactCache.hasAllKeys(list.map(key => new URL(key.dataPath, tensorCacheUrl).href));
                // `loading`: we have finished downloading (or already cacheOnly) and are loading onto WebGPU
                const reportCallback = (iter, loading = false) => {
                    // report
                    for (let j = 0; j < this.initProgressCallback.length; ++j) {
                        let text;
                        if (loading) {
                            text = "Loading model from cache[" + iter + "/" + list.length + "]: ";
                            text += Math.ceil(fetchedBytes / (1024 * 1024)).toString() + "MB loaded. ";
                            text += Math.floor(fetchedBytes * 100 / totalBytes).toString() + "% completed, ";
                            text += timeElapsed + " secs elapsed.";
                        }
                        else {
                            text = "Fetching param cache[" + iter + "/" + list.length + "]: ";
                            text += Math.ceil(fetchedBytes / (1024 * 1024)).toString() + "MB fetched. ";
                            text += Math.floor(fetchedBytes * 100 / totalBytes).toString() + "% completed, ";
                            text += timeElapsed + " secs elapsed.";
                            text += " It can take a while when we first visit this page to populate the cache.";
                            text += " Later refreshes will become faster.";
                        }
                        this.initProgressCallback[j]({
                            progress: fetchedBytes / totalBytes,
                            timeElapsed: timeElapsed,
                            text: text
                        });
                    }
                };
                for (let j = 0; j < this.initProgressCallback.length; ++j) {
                    this.initProgressCallback[j]({
                        progress: fetchedBytes / totalBytes,
                        timeElapsed: 0,
                        text: "Start to fetch params",
                    });
                }
                // First download all shards to cache parallely if not yet in cache
                const downloadCache = (start, end) => __awaiter(this, void 0, void 0, function* () {
                    // Download params [start, end) from `list`
                    for (let i = start; i < end; i++) {
                        const shard = list[i];
                        const dataUrl = new URL(shard.dataPath, tensorCacheUrl).href;
                        try {
                            yield artifactCache.addToCache(dataUrl, "arraybuffer", signal);
                        }
                        catch (err) {
                            this.env.logger("Error: Cannot fetch " + dataUrl + " err= " + err);
                            throw err;
                        }
                        timeElapsed = Math.ceil((perf.now() - tstart) / 1000);
                        fetchedBytes += shard.nbytes;
                        reportCallback(++fetchedShards, /*loading=*/ false);
                    }
                });
                // We launch 4 parallel for loops to limit the max concurrency to 4 download
                if (!cacheOnly) {
                    const loopSize = Math.floor(list.length / 4);
                    yield Promise.all([
                        downloadCache(0, loopSize),
                        downloadCache(loopSize, 2 * loopSize),
                        downloadCache(2 * loopSize, 3 * loopSize),
                        downloadCache(3 * loopSize, list.length)
                    ]);
                }
                // Reset for the loading phase to avoid double counting with download phase
                fetchedBytes = 0;
                fetchedShards = 0;
                // Then iteratively, load the shard from cache
                for (let i = 0; i < list.length; ++i) {
                    const shard = list[i];
                    const dataUrl = new URL(shard.dataPath, tensorCacheUrl).href;
                    let buffer;
                    try {
                        buffer = yield artifactCache.fetchWithCache(dataUrl, "arraybuffer");
                    }
                    catch (err) {
                        this.env.logger("Error: Cannot fetch " + dataUrl + " err= " + err);
                        throw err;
                    }
                    const shardRecords = shard.records;
                    for (let j = 0; j < shardRecords.length; ++j) {
                        try {
                            const rec = shardRecords[j];
                            const cpu_arr = this.withNewScope(() => {
                                return this.detachFromCurrentScope(this.empty(rec.shape, rec.dtype, this.cpu()));
                            });
                            const recSource = buffer.slice(rec.byteOffset, rec.byteOffset + rec.nbytes);
                            // first sync copy to cpu.
                            this.ctx.arrayDecodeStorage(cpu_arr, new Uint8Array(recSource), rec.format, rec.dtype);
                            // then async stream into GPU if needed
                            if (device.deviceType === DeviceStrToEnum.cpu) {
                                this.tensorCacheUpdate(rec.name, cpu_arr, false);
                                cpu_arr.dispose();
                            }
                            else {
                                // allocate a gpu arr and async copy to it.
                                const gpu_arr = this.withNewScope(() => {
                                    return this.detachFromCurrentScope(this.empty(rec.shape, rec.dtype, device));
                                });
                                gpu_arr.copyFrom(cpu_arr);
                                yield device.sync();
                                this.tensorCacheUpdate(rec.name, gpu_arr, false);
                                cpu_arr.dispose();
                                gpu_arr.dispose();
                            }
                        }
                        catch (err) {
                            this.env.logger("Failed to load shard " + i + "'s record: " + JSON.stringify(shardRecords[j]) + "\n" +
                                "Error: " + err);
                            throw err;
                        }
                    }
                    fetchedBytes += shard.nbytes;
                    timeElapsed = Math.ceil((perf.now() - tstart) / 1000);
                    reportCallback(++fetchedShards, /*loading=*/ true);
                }
            });
        }
        /**
         * Create a new {@link Scalar} that can be passed to a PackedFunc.
         * @param value The number value.
         * @param dtype The dtype string.
         * @returns The created scalar.
         */
        scalar(value, dtype) {
            return new Scalar(value, dtype);
        }
        /**
         * Create a new {@link DLDevice}
         * @param deviceType The device type.
         * @param deviceId The device index.
         * @returns The created device.
         */
        device(deviceType, deviceId = 0) {
            return new DLDevice(deviceType, deviceId, this.lib);
        }
        /**
         * Create a new cpu {@link DLDevice}
         * @param deviceId The device index.
         */
        cpu(deviceId = 0) {
            return this.device("cpu", deviceId);
        }
        /**
         * Create a new webgpu {@link DLDevice}
         * @param deviceId The device index.
         */
        webgpu(deviceId = 0) {
            return this.device("webgpu", deviceId);
        }
        /**
         * Create an empty {@link Tensor} with given shape and dtype.
         *
         * @param shape The shape of the array.
         * @param dtype The data type of the array.
         * @param dev The device of the ndarray.
         * @returns The created ndarray.
         */
        empty(shape, dtype = "float32", dev = this.device("cpu", 0)) {
            shape = typeof shape === "number" ? [shape] : shape;
            return this.ctx.tensorEmpty(this.makeShapeTuple(shape), dtype, dev, null);
        }
        /**
         * Create am uniform {@link Tensor} with given shape.
         *
         * @param shape The shape of the array.
         * @param low The low value.
         * @param high The high value.
         * @param dev The device of the ndarray.
         * @returns The created ndarray.
         */
        uniform(shape, low, high, dev) {
            const ret = this.empty(shape, "float32", dev);
            const size = shape.reduce((a, b) => {
                return a * b;
            }, 1);
            const scale = high - low;
            const input = new Float32Array(size);
            for (let i = 0; i < input.length; ++i) {
                input[i] = low + this.rng.randomFloat() * scale;
            }
            return ret.copyFrom(input);
        }
        /**
         * Set the seed of the internal LinearCongruentialGenerator.
         */
        setSeed(seed) {
            this.rng.setSeed(seed);
        }
        /**
         * Sample index via top-p sampling.
         *
         * @param logits The input logits before normalization.
         * @param temperature  The temperature factor, will take argmax if temperature = 0.0
         * @param top_p The top_p
         * @returns The sampled index.
         */
        sampleTopPFromLogits(logits, temperature, top_p) {
            return this.ctx.sampleTopPFromLogits(logits, temperature, top_p, this.rng.randomFloat());
        }
        /**
         * Sample index via top-p sampling.
         *
         * @param prob The distribution, i.e. logits after `applySoftmaxWithTemperature()` is performed.
         * @param top_p The top_p
         * @returns The sampled index.
         */
        sampleTopPFromProb(prob, top_p) {
            return this.ctx.sampleTopPFromProb(prob, top_p, this.rng.randomFloat());
        }
        /**
         * Apply repetition penalty to the logits.
         * @param logits The input logits before penalty.
         * @param token_ids The appeared token ids.
         * @param penalty The penalty factor.
         */
        applyRepetitionPenalty(logits, token_ids, penalty) {
            return this.ctx.applyRepetitionPenalty(logits, token_ids, penalty);
        }
        /**
         * Apply presence and frequency penalty. This is an inplace operation.
         * @param logits The input logits before penalty.
         * @param token_ids The appeared token ids.
         * @param token_freqs The number of times each token has appeared since last PrefillStep.
         * token_freqs[i] is the frequency of token_ids[i], for all i. And all token_freqs should be >= 1.
         * @param presence_penalty The penalty factor.
         * @param frequency_penalty The penalty factor.
         */
        applyPresenceAndFrequencyPenalty(logits, token_ids, token_freqs, presence_penalty, frequency_penalty) {
            return this.ctx.applyPresenceAndFrequencyPenalty(logits, token_ids, token_freqs, presence_penalty, frequency_penalty);
        }
        /**
         * Apply softmax with temperature to the logits.
         * @param logits The input logits before softmax w/ temperature.
         * @param temperature The temperature factor.
         */
        applySoftmaxWithTemperature(logits, temperature) {
            return this.ctx.applySoftmaxWithTemperature(logits, temperature);
        }
        /**
         * Bind canvas to the current WebGPU context
         * @param canvas The canvas.
         */
        bindCanvas(canvas) {
            var _a;
            (_a = this.lib.webGPUContext) === null || _a === void 0 ? void 0 : _a.bindCanvas(canvas);
        }
        /**
         * Show image in canvas.
         *
         * @param dataRGBA Image array in height x width uint32 Tensor RGBA format on GPU.
         */
        showImage(dataRGBA) {
            var _a;
            if (dataRGBA.shape.length != 2) {
                throw Error("Require a height x width uint32 Tensor in RGBA" +
                    "get shape=" + dataRGBA.shape.toString() + " instead.");
            }
            if (dataRGBA.device.deviceType != DeviceStrToEnum.webgpu) {
                throw new Error("Can only run showImage on WebGPU array, " +
                    "get " + DeviceEnumToStr[dataRGBA.device.deviceType] + " instead.");
            }
            if (dataRGBA.dtype != "uint32") {
                throw Error("Require a height x width uint32 Tensor in RGBA, " +
                    "get " + dataRGBA.dtype + " instead.");
            }
            (_a = this.lib.webGPUContext) === null || _a === void 0 ? void 0 : _a.drawImageFromBuffer(dataRGBA.getDataPtr(), dataRGBA.shape[0], dataRGBA.shape[1]);
        }
        /**
         * Clear canvas
         */
        clearCanvas() {
            var _a;
            (_a = this.lib.webGPUContext) === null || _a === void 0 ? void 0 : _a.clearCanvas();
        }
        /**
         * Create an tuple {@link TVMArray} input array.
         *
         * The input array can be passed to tvm runtime function
         * and needs to b explicitly disposed.
         *
         * @param inputs The input array
         * @returns The result array.
         */
        makeTVMArray(inputs) {
            const CALL_STACK_LIMIT = 30000;
            const inputsLength = inputs.length;
            if (inputsLength <= CALL_STACK_LIMIT) {
                return this.ctx.arrayMake(...inputs);
            }
            // If too many elements, TypeScript would complain `Maximum call stack size exceeded`
            // So we make several arrays and concatenate them
            const listOfArrays = [];
            for (let begin = 0; begin < inputsLength; begin += CALL_STACK_LIMIT) {
                const end = Math.min(inputsLength, begin + CALL_STACK_LIMIT);
                const chunk = inputs.slice(begin, end);
                listOfArrays.push(this.ctx.arrayMake(...chunk));
            }
            return this.ctx.arrayConcat(...listOfArrays);
        }
        /**
         * Join a sequence of Tensors that represent embeddings.
          * @param embeddings A list of embeddings in Tensors, each array i has shape (m_i, hidden_size).
         * @returns An Tensor of shape (\sum_{i} {m}, hidden_size)
         */
        concatEmbeddings(embeddings) {
            // 1. Check shape validity
            const hidden_size = embeddings[0].shape[1];
            embeddings.forEach((input) => {
                if (input.shape.length !== 2 || input.shape[1] !== hidden_size) {
                    throw new Error("Expect embeddings to concatenate have shape (m_i, hidden_size).");
                }
            });
            // 2. Call global func
            if (this.ctx.concatEmbeddings === undefined) {
                throw new Error("Global function tvmjs.runtime.ConcatEmbeddings was " +
                    "not found, but called concatEmbeddings.");
            }
            return this.ctx.concatEmbeddings(...embeddings);
        }
        /**
         * Create a shape tuple to pass to runtime.
         * @param shape The shape .
         * @returns The created shape tuple.
         */
        makeShapeTuple(shape) {
            const key = CacheState.computeShapeKey(shape);
            return this.cacheState.shapeCache.get(key, () => {
                const shapeArray = shape.map((value) => new Scalar(value, "int"));
                const tuple = this.ctx.makeShapeTuple(...shapeArray);
                // Detach from scope so the cached object survives across scopes.
                this.detachFromCurrentScope(tuple);
                return tuple;
            });
        }
        /**
         * Get type index from type key.
         * @param typeKey The type key.
         * @returns The corresponding type index.
         */
        typeKey2Index(typeKey) {
            const stack = this.lib.getOrAllocCallStack();
            const typeKeyOffset = stack.allocByteArrayForString(typeKey);
            const outOffset = stack.allocPtrArray(1);
            const outPtr = stack.ptrFromOffset(outOffset);
            stack.commitToWasmMemory(outOffset);
            this.lib.checkCall(this.lib.exports.TVMFFITypeKeyToIndex(stack.ptrFromOffset(typeKeyOffset), outPtr));
            const typeIndex = this.memory.loadU32(outPtr);
            this.lib.recycleCallStack(stack);
            return typeIndex;
        }
        /**
         * Register an object constructor.
         * @param typeKey The name of the function.
         * @param func Function to be registered.
         * @param override Whether overwrite function in existing registry.
         */
        registerObjectConstructor(typeKey, func, override = false) {
            const typeIndex = this.typeKey2Index(typeKey);
            if (this.objFactory.has(typeIndex)) {
                if (!override) {
                    throw new Error("Type " + typeKey + " already registered");
                }
            }
            this.objFactory.set(typeIndex, func);
        }
        /**
         * Wrap a function obtained from tvm runtime as AsyncPackedFunc
         * through the asyncify mechanism
         *
         * You only need to call it if the function may contain callback into async
         * JS function via asynctify. A common one can be GPU synchronize.
         *
         * It is always safe to wrap any function as Asynctify, however you do need
         * to make sure you use await when calling the funciton.
         *
         * @param func The PackedFunc.
         * @returns The wrapped AsyncPackedFunc
         */
        wrapAsyncifyPackedFunc(func) {
            const asyncFunc = this.asyncifyHandler.wrapExport(func);
            asyncFunc.dispose = func.dispose;
            asyncFunc._tvmPackedCell = func._tvmPackedCell;
            return asyncFunc;
        }
        /**
         * Register async function as asynctify callable in global environment.
         *
         * @param name The name of the function.
         * @param func function to be registered.
         * @param override Whether overwrite function in existing registry.
         *
          * Note: This function is handled via asynctify mechanism.
         * The wasm needs to be compiled with Asynctify
         */
        registerAsyncifyFunc(name, func, override = false) {
            const asyncWrapped = this.asyncifyHandler.wrapImport(func);
            this.registerFunc(name, asyncWrapped, override);
        }
        /**
         * Register an asyncfunction to be global function in the server.
         *
         * @param name The name of the function.
         * @param func function to be registered.
         * @param override Whether overwrite function in existing registry.
         *
          * Note: The async function will only be used for serving remote calls in the rpc.
         * These functions contains explicit continuation
         */
        registerAsyncServerFunc(name, func, override = false) {
            const asyncVariant = (...args) => {
                const fargs = args.slice(0, args.length - 1);
                // need to keep it alive until callback is fulfilled.
                const callback = this.detachFromCurrentScope(args[args.length - 1]);
                const promise = func(...fargs);
                const onFulfilled = (rv) => {
                    callback(this.scalar(AsyncCallbackCode.kReturn, "int32"), rv);
                    callback.dispose();
                };
                const onRejected = (reason) => {
                    callback(this.scalar(AsyncCallbackCode.kException, "int32"), reason.toString());
                    callback.dispose();
                };
                promise.then(onFulfilled, onRejected);
            };
            this.registerFunc("__async." + name, asyncVariant, override);
        }
        /**
         * Asynchronously load webgpu pipelines when possible.
         * @param mod The input module.
         */
        asyncLoadWebGPUPipelines(mod) {
            return __awaiter(this, void 0, void 0, function* () {
                if (this.lib.webGPUContext === undefined)
                    throw Error("WebGPU not initialied");
                const webgpuContext = this.lib.webGPUContext;
                this.beginScope();
                const fmap_str = mod.getFunction("webgpu.get_fmap", true)();
                const fmap = JSON.parse(fmap_str);
                const fGetShader = this.detachFromCurrentScope(mod.getFunction("webgpu.get_shader"));
                const fUpdatePrebuild = this.detachFromCurrentScope(mod.getFunction("webgpu.update_prebuild"));
                this.endScope();
                const perf = getPerformance();
                const tstart = perf.now();
                let tlastReport = tstart;
                let finishCounter = 0;
                const fmapEntries = Object.entries(fmap);
                let allEvents = Promise.resolve();
                for (const [key, finfo] of fmapEntries) {
                    const code = fGetShader(key);
                    assert(key === finfo.name);
                    const event = webgpuContext.createShaderAsync(finfo, code).then((func) => {
                        this.beginScope();
                        fUpdatePrebuild(key, func);
                        this.endScope();
                    }).then(() => {
                        finishCounter += 1;
                        const tend = perf.now();
                        // skip report if gap is smaller than 1000
                        if ((tend - tlastReport) < 1000 && finishCounter != fmapEntries.length) {
                            return;
                        }
                        tlastReport = tend;
                        const timeElapsed = Math.ceil((perf.now() - tstart) / 1000);
                        // report
                        for (let j = 0; j < this.initProgressCallback.length; ++j) {
                            const progress = finishCounter / fmapEntries.length;
                            let text = "Loading GPU shader modules[" + finishCounter + "/" + fmapEntries.length + "]: ";
                            text += Math.floor(progress * 100).toString() + "% completed, ";
                            text += timeElapsed + " secs elapsed.";
                            this.initProgressCallback[j]({
                                progress: progress,
                                timeElapsed: timeElapsed,
                                text: text
                            });
                        }
                    });
                    allEvents = Promise.all([allEvents, event]).then(() => { });
                }
                yield allEvents;
                assert(finishCounter === fmapEntries.length);
            });
        }
        /**
         * Initialize webgpu in the runtime.
         * @param device The given GPU device.
         */
        initWebGPU(device) {
            device.addEventListener("uncapturederror", (event) => {
                console.error("A WebGPU error was not captured: ", event);
            });
            device.lost.then((info) => {
                if (this.deviceLostIsError) {
                    console.error("Device lost, calling Instance.dispose(). Please initialize again. ", info);
                    this.dispose();
                }
            });
            this.deviceLostIsError = true;
            const webGPUContext = new WebGPUContext(this.memory, device);
            this.registerFunc("wasm.WebGPUDeviceAPI", (name) => {
                return webGPUContext.getDeviceAPI(name);
            });
            this.registerFunc("wasm.WebGPUCreateShader", (info, code) => {
                const finfo = JSON.parse(info);
                return webGPUContext.createShader(finfo, code);
            });
            this.registerAsyncServerFunc("wasm.WebGPUWaitForTasks", () => __awaiter(this, void 0, void 0, function* () {
                yield webGPUContext.sync();
            }));
            if (this.asyncifyHandler.enabled()) {
                this.registerAsyncifyFunc("__asyncify.WebGPUWaitForTasks", () => __awaiter(this, void 0, void 0, function* () {
                    yield webGPUContext.sync();
                }));
            }
            this.lib.webGPUContext = webGPUContext;
        }
        /** Register all object factory */
        registerObjectFactoryFuncs() {
            this.registerObjectConstructor("ffi.Array", (handle, lib, ctx) => {
                return new TVMArray(handle, lib, ctx);
            });
            this.registerObjectConstructor("ffi.Module", (handle, lib, ctx) => {
                return new Module(handle, lib, ctx);
            });
        }
        /** Register global packed functions needed by the backend to the env. */
        registerEnvGlobalPackedFuncs() {
            // Register the timer function to enable the time_evaluator.
            const perf = getPerformance();
            // Helper function to time the finvoke
            const timeExecution = (finvoke, dev, nstep, repeat, minRepeatMs, limitZeroTimeIterations, cooldownIntervalMs, repeatsToCooldown) => __awaiter(this, void 0, void 0, function* () {
                // detach and explicit dispose when tasks is fullfilled
                // the promise will immediately return and we need to makesure
                // finvoke do not get recycled.
                this.ctx.detachFromCurrentScope(finvoke);
                finvoke(this.scalar(1, "int32"));
                yield dev.sync();
                const result = [];
                let setupNumber = nstep;
                for (let i = 0; i < repeat; ++i) {
                    let durationMs = 0.0;
                    let absoluteZeroTimes = 0;
                    do {
                        if (durationMs > 0.0) {
                            const golden_ratio = 1.618;
                            setupNumber = Math.floor(Math.max(minRepeatMs / (durationMs / setupNumber) + 1, setupNumber * golden_ratio));
                        }
                        const tstart = perf.now();
                        finvoke(this.scalar(setupNumber, "int32"));
                        yield dev.sync();
                        const tend = perf.now();
                        durationMs = tend - tstart;
                        if (durationMs === 0) {
                            absoluteZeroTimes++;
                        }
                    } while (durationMs < minRepeatMs && absoluteZeroTimes < limitZeroTimeIterations);
                    const speed = durationMs / setupNumber / 1000;
                    result.push(speed);
                    if (cooldownIntervalMs > 0.0 && (i % repeatsToCooldown) === 0) {
                        yield new Promise(r => setTimeout(r, cooldownIntervalMs));
                    }
                }
                const ret = new Float64Array(result.length);
                ret.set(result);
                // dispose finvoke
                finvoke.dispose();
                return new Uint8Array(ret.buffer);
            });
            const addOne = (x) => __awaiter(this, void 0, void 0, function* () {
                yield new Promise(resolve => setTimeout(resolve, 100));
                return x + 1;
            });
            this.registerAsyncServerFunc("wasm.TimeExecution", timeExecution);
            this.registerAsyncServerFunc("testing.asyncAddOne", addOne);
        }
        createPackedFuncFromSafeCallType(func) {
            let findex = this.env.packedCFuncTable.length;
            if (this.env.packedCFuncTableFreeId.length != 0) {
                findex = this.env.packedCFuncTableFreeId.pop();
            }
            else {
                this.env.packedCFuncTable.push(undefined);
            }
            this.env.packedCFuncTable[findex] = func;
            const stack = this.lib.getOrAllocCallStack();
            const outOffset = stack.allocPtrArray(1);
            const outPtr = stack.ptrFromOffset(outOffset);
            this.lib.checkCall(this.exports
                .TVMFFIWasmFunctionCreate(findex, outPtr));
            const ret = this.makePackedFunc(this.memory.loadPointer(outPtr));
            this.lib.recycleCallStack(stack);
            return ret;
        }
        /**
         * Set packed function arguments into the location indicated by argsValue and argsCode.
         * Allocate new temporary space from the stack if necessary.
         *
          * @param stack The call stack.
         * @param args  The input arguments.
         * @param packedArgs The offset of packedArgs.
         */
        setPackedArguments(stack, args, packedArgs) {
            for (let i = 0; i < args.length; ++i) {
                let val = args[i];
                const tp = typeof val;
                const argOffset = packedArgs + i * 16 /* SizeOf.TVMFFIAny */;
                const argTypeIndexOffset = argOffset;
                const argZeroPaddingOffset = argOffset + 4 /* SizeOf.I32 */;
                const argValueOffset = argOffset + 4 /* SizeOf.I32 */ * 2;
                // Convert string[] to a TVMArray of, hence treated as a TVMObject
                if (val instanceof Array && val.every(e => typeof e === "string")) {
                    const tvmStringArray = [];
                    val.forEach(e => { tvmStringArray.push(e); });
                    val = this.makeTVMArray(tvmStringArray);
                }
                // clear off the extra zero padding before ptr storage
                stack.storeI32(argZeroPaddingOffset, 0);
                // clear off the extra zero padding after ptr storage
                stack.storeI32(argValueOffset + 4 /* SizeOf.I32 */, 0);
                if (val instanceof Tensor) {
                    if (!val.isView) {
                        stack.storeI32(argTypeIndexOffset, 70 /* TypeIndex.kTVMFFITensor */);
                        stack.storePtr(argValueOffset, val.getHandle());
                    }
                    else {
                        stack.storeI32(argTypeIndexOffset, 7 /* TypeIndex.kTVMFFIDLTensorPtr */);
                        stack.storePtr(argValueOffset, val.getHandle());
                    }
                }
                else if (val instanceof Scalar) {
                    if (val.dtype.startsWith("int") || val.dtype.startsWith("uint")) {
                        stack.storeI32(argTypeIndexOffset, 1 /* TypeIndex.kTVMFFIInt */);
                        stack.storeI64(argValueOffset, val.value);
                    }
                    else if (val.dtype.startsWith("float")) {
                        stack.storeI32(argTypeIndexOffset, 3 /* TypeIndex.kTVMFFIFloat */);
                        stack.storeF64(argValueOffset, val.value);
                    }
                    else {
                        assert(val.dtype === "handle", "Expect handle");
                        stack.storeI32(argTypeIndexOffset, 4 /* TypeIndex.kTVMFFIOpaquePtr */);
                        stack.storePtr(argValueOffset, val.value);
                    }
                }
                else if (val instanceof DLDevice) {
                    stack.storeI32(argTypeIndexOffset, 6 /* TypeIndex.kTVMFFIDevice */);
                    stack.storeI32(argValueOffset, val.deviceType);
                    stack.storeI32(argValueOffset + 4 /* SizeOf.I32 */, val.deviceId);
                }
                else if (tp === "boolean") {
                    stack.storeI32(argTypeIndexOffset, 2 /* TypeIndex.kTVMFFIBool */);
                    stack.storeI64(argValueOffset, val ? 1 : 0);
                }
                else if (tp === "number") {
                    stack.storeI32(argTypeIndexOffset, 3 /* TypeIndex.kTVMFFIFloat */);
                    stack.storeF64(argValueOffset, val);
                }
                else if (tp === "function" && val.hasOwnProperty("_tvmPackedCell")) {
                    stack.storePtr(argValueOffset, val._tvmPackedCell.getHandle());
                    stack.storeI32(argTypeIndexOffset, 68 /* TypeIndex.kTVMFFIFunction */);
                }
                else if (val === null || val === undefined) {
                    stack.storeI32(argTypeIndexOffset, 0 /* TypeIndex.kTVMFFINone */);
                    stack.storePtr(argValueOffset, 0);
                }
                else if (tp === "string") {
                    stack.storeI32(argTypeIndexOffset, 8 /* TypeIndex.kTVMFFIRawStr */);
                    stack.allocThenSetArgString(argValueOffset, val);
                }
                else if (val instanceof Uint8Array) {
                    stack.storeI32(argTypeIndexOffset, 9 /* TypeIndex.kTVMFFIByteArrayPtr */);
                    stack.allocThenSetArgBytes(argValueOffset, val);
                }
                else if (val instanceof Function) {
                    val = this.toPackedFuncInternal(val, false);
                    stack.tempArgs.push(val);
                    stack.storeI32(argTypeIndexOffset, 68 /* TypeIndex.kTVMFFIFunction */);
                    stack.storePtr(argValueOffset, val._tvmPackedCell.getHandle());
                }
                else if (val instanceof Module) {
                    stack.storeI32(argTypeIndexOffset, 73 /* TypeIndex.kTVMFFIModule */);
                    stack.storePtr(argValueOffset, val.getHandle());
                }
                else if (val instanceof TVMObject) {
                    stack.storeI32(argTypeIndexOffset, val.typeIndex());
                    stack.storePtr(argValueOffset, val.getHandle());
                }
                else {
                    throw new Error("Unsupported argument type " + tp + " value=`" + val.toString() + "`");
                }
            }
        }
        wrapJSFuncAsSafeCallType(func) {
            const lib = this.lib;
            return (self, packedArgs, numArgs, ret) => {
                const jsArgs = [];
                // use scope to track js values.
                this.ctx.beginScope();
                for (let i = 0; i < numArgs; ++i) {
                    const argPtr = packedArgs + i * 16 /* SizeOf.TVMFFIAny */;
                    const typeIndex = lib.memory.loadI32(argPtr);
                    if (typeIndex >= 8 /* TypeIndex.kTVMFFIRawStr */) {
                        // NOTE: the following code have limitations in asyncify mode.
                        // The reason is that the TVMFFIAnyViewToOwnedAny will simply
                        // get skipped during the rewinding process, causing memory failure
                        if (!this.asyncifyHandler.isNormalStackState()) {
                            throw Error("Cannot handle str/object argument callback in asyncify mode");
                        }
                        lib.checkCall(lib.exports.TVMFFIAnyViewToOwnedAny(argPtr, argPtr));
                    }
                    jsArgs.push(this.retValueToJS(argPtr, true));
                }
                let rv;
                try {
                    rv = func(...jsArgs);
                }
                catch (error) {
                    // error handling
                    // store error via SetLastError
                    this.ctx.endScope();
                    const errKind = "JSCallbackError";
                    const errMsg = error.message;
                    const stack = lib.getOrAllocCallStack();
                    const errKindOffset = stack.allocRawBytes(errKind.length + 1);
                    stack.storeRawBytes(errKindOffset, StringToUint8Array(errKind));
                    const errMsgOffset = stack.allocRawBytes(errMsg.length + 1);
                    stack.storeRawBytes(errMsgOffset, StringToUint8Array(errMsg));
                    stack.commitToWasmMemory();
                    this.lib.exports.TVMFFIErrorSetRaisedFromCStr(stack.ptrFromOffset(errKindOffset), stack.ptrFromOffset(errMsgOffset));
                    this.lib.recycleCallStack(stack);
                    return -1;
                }
                // normal return path
                // recycle all js object value in function unless we want to retain them.
                this.ctx.endScope();
                if (rv !== undefined && rv !== null) {
                    const stack = lib.getOrAllocCallStack();
                    const argOffset = stack.allocRawBytes(16 /* SizeOf.TVMFFIAny */);
                    this.setPackedArguments(stack, [rv], argOffset);
                    stack.commitToWasmMemory();
                    const argPtr = stack.ptrFromOffset(argOffset);
                    lib.checkCall(lib.exports.TVMFFIAnyViewToOwnedAny(argPtr, ret));
                    lib.recycleCallStack(stack);
                }
                return 0;
            };
        }
        makePackedFunc(handle) {
            const cell = new PackedFuncCell(handle, this.lib, this.ctx);
            const packedFunc = (...args) => {
                const stack = this.lib.getOrAllocCallStack();
                const argsOffset = stack.allocRawBytes(16 /* SizeOf.TVMFFIAny */ * args.length);
                this.setPackedArguments(stack, args, argsOffset);
                const retOffset = stack.allocRawBytes(16 /* SizeOf.TVMFFIAny */);
                // pre-store the result to be null
                stack.storeI32(retOffset, 0 /* TypeIndex.kTVMFFINone */);
                // clear off the extra zero padding before ptr storage
                stack.storeI32(retOffset + 4 /* SizeOf.I32 */, 0);
                stack.commitToWasmMemory();
                this.lib.checkCall(this.exports.TVMFFIFunctionCall(cell.getHandle(), stack.ptrFromOffset(argsOffset), args.length, stack.ptrFromOffset(retOffset)));
                const ret = this.retValueToJS(stack.ptrFromOffset(retOffset), false);
                this.lib.recycleCallStack(stack);
                return ret;
            };
            // Attach attributes to the function type.
            // This is because javascript do not allow us to overload call.
            const ret = packedFunc;
            ret.dispose = () => {
                cell.dispose();
            };
            ret._tvmPackedCell = cell;
            return ret;
        }
        /**
         * Creaye return value of the packed func. The value us auto-tracked for dispose.
         * @param resultAnyPtr The location of rvalue
         * @param callbackArg Whether it is being used in callbackArg.
         * @returns The JS value.
         */
        retValueToJS(resultAnyPtr, callbackArg) {
            const typeIndex = this.memory.loadI32(resultAnyPtr);
            const valuePtr = resultAnyPtr + 4 /* SizeOf.I32 */ * 2;
            switch (typeIndex) {
                case 0 /* TypeIndex.kTVMFFINone */: return undefined;
                case 2 /* TypeIndex.kTVMFFIBool */:
                    return this.memory.loadI64(valuePtr) != 0;
                case 1 /* TypeIndex.kTVMFFIInt */:
                    return this.memory.loadI64(valuePtr);
                case 3 /* TypeIndex.kTVMFFIFloat */:
                    return this.memory.loadF64(valuePtr);
                case 4 /* TypeIndex.kTVMFFIOpaquePtr */: {
                    return this.memory.loadPointer(valuePtr);
                }
                case 70 /* TypeIndex.kTVMFFITensor */: {
                    return this.ctx.attachToCurrentScope(new Tensor(this.memory.loadPointer(valuePtr), this.lib, this.ctx, false));
                }
                case 7 /* TypeIndex.kTVMFFIDLTensorPtr */: {
                    assert(callbackArg);
                    // no need to attach as we are only looking at view
                    return new Tensor(this.memory.loadPointer(valuePtr), this.lib, this.ctx, true);
                }
                case 68 /* TypeIndex.kTVMFFIFunction */: {
                    return this.ctx.attachToCurrentScope(this.makePackedFunc(this.memory.loadPointer(valuePtr)));
                }
                case 6 /* TypeIndex.kTVMFFIDevice */: {
                    const deviceType = this.memory.loadI32(valuePtr);
                    const deviceId = this.memory.loadI32(valuePtr + 4 /* SizeOf.I32 */);
                    return this.device(deviceType, deviceId);
                }
                case 5 /* TypeIndex.kTVMFFIDataType */: {
                    // simply return dtype as tring to keep things simple
                    this.lib.checkCall(this.lib.exports.TVMFFIDataTypeToString(valuePtr, valuePtr));
                    const strObjPtr = this.memory.loadPointer(valuePtr);
                    const result = this.memory.loadByteArrayAsString(strObjPtr + 24 /* SizeOf.ObjectHeader */);
                    this.lib.checkCall(this.lib.exports.TVMFFIObjectDecRef(strObjPtr));
                    return result;
                }
                case 11 /* TypeIndex.kTVMFFISmallStr */: {
                    return this.memory.loadSmallStr(resultAnyPtr);
                }
                case 65 /* TypeIndex.kTVMFFIStr */: {
                    const strObjPtr = this.memory.loadPointer(valuePtr);
                    const result = this.memory.loadByteArrayAsString(strObjPtr + 24 /* SizeOf.ObjectHeader */);
                    this.lib.checkCall(this.lib.exports.TVMFFIObjectDecRef(strObjPtr));
                    return result;
                }
                case 12 /* TypeIndex.kTVMFFISmallBytes */: {
                    return this.memory.loadSmallBytes(resultAnyPtr);
                }
                case 66 /* TypeIndex.kTVMFFIBytes */: {
                    const bytesObjPtr = this.memory.loadPointer(valuePtr);
                    const result = this.memory.loadByteArrayAsBytes(bytesObjPtr + 24 /* SizeOf.ObjectHeader */);
                    this.lib.checkCall(this.lib.exports.TVMFFIObjectDecRef(bytesObjPtr));
                    return result;
                }
                default: {
                    if (typeIndex >= 64 /* TypeIndex.kTVMFFIStaticObjectBegin */) {
                        const obj = new TVMObject(this.memory.loadPointer(valuePtr), this.lib, this.ctx);
                        const func = this.objFactory.get(obj.typeIndex());
                        if (func != undefined) {
                            return this.ctx.attachToCurrentScope(func(obj.getHandle(), this.lib, this.ctx));
                        }
                        else {
                            return this.ctx.attachToCurrentScope(obj);
                        }
                    }
                    else {
                        throw new Error("Unsupported return type code=" + typeIndex);
                    }
                }
            }
        }
    }
    /**
     * Asynchrously instantiate a new {@link Instance}.
     *
     * importObject can also be a {@link LibraryProvider} object,
     * a WASI object, or an object containing wasmLibraryProvider field.
     * We can take benefit of syslib implementations from the Emscripten
     * by passing its generated js Module as the imports.
     *
     * @param bufferSource The source to be compiled.
     * @param importObject The import objects.
     * @param logger The system logger.
     */
    function instantiate(bufferSource, importObject = {}, logger = console.log) {
        const env = new Environment(importObject, logger);
        return WebAssembly.instantiate(bufferSource, env.imports).then((result) => {
            return new Instance(result.module, {}, result.instance, env);
        });
    }

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
    var RPCServerState;
    (function (RPCServerState) {
        RPCServerState[RPCServerState["InitHeader"] = 0] = "InitHeader";
        RPCServerState[RPCServerState["InitHeaderKey"] = 1] = "InitHeaderKey";
        RPCServerState[RPCServerState["InitServer"] = 2] = "InitServer";
        RPCServerState[RPCServerState["WaitForCallback"] = 3] = "WaitForCallback";
        RPCServerState[RPCServerState["ReceivePacketHeader"] = 4] = "ReceivePacketHeader";
        RPCServerState[RPCServerState["ReceivePacketBody"] = 5] = "ReceivePacketBody";
    })(RPCServerState || (RPCServerState = {}));
    /** RPC magic header */
    const RPC_MAGIC = 0xff271;
    /**
     * An utility class to read from binary bytes.
     */
    class ByteStreamReader {
        constructor(bytes) {
            this.offset = 0;
            this.bytes = bytes;
        }
        readU32() {
            const i = this.offset;
            const b = this.bytes;
            const val = b[i] | (b[i + 1] << 8) | (b[i + 2] << 16) | (b[i + 3] << 24);
            this.offset += 4;
            return val;
        }
        readU64() {
            const val = this.readU32();
            this.offset += 4;
            return val;
        }
        readByteArray() {
            const len = this.readU64();
            assert(this.offset + len <= this.bytes.byteLength);
            const ret = new Uint8Array(len);
            ret.set(this.bytes.slice(this.offset, this.offset + len));
            this.offset += len;
            return ret;
        }
    }
    /**
     * A websocket based RPC
     */
    class RPCServer {
        constructor(url, key, getImports, logger = console.log, tensorCacheUrl = "", tensorCacheDevice = "cpu", initProgressCallback = undefined, asyncOnServerLoad = undefined) {
            this.state = RPCServerState.InitHeader;
            this.pendingSend = Promise.resolve();
            this.inst = undefined;
            this.globalObjects = [];
            this.currPacketLength = 0;
            this.remoteKeyLength = 0;
            this.pendingBytes = 0;
            this.buffredBytes = 0;
            this.messageQueue = [];
            this.url = url;
            this.key = key;
            this.name = "WebSocketRPCServer[" + this.key + "]: ";
            this.getImports = getImports;
            this.logger = logger;
            this.tensorCacheUrl = tensorCacheUrl;
            this.tensorCacheDevice = tensorCacheDevice;
            this.initProgressCallback = initProgressCallback;
            this.asyncOnServerLoad = asyncOnServerLoad;
            this.checkLittleEndian();
            this.socket = createWebSocket(url);
            this.socket.binaryType = "arraybuffer";
            this.socket.addEventListener("open", (event) => {
                return this.onOpen(event);
            });
            this.socket.addEventListener("message", (event) => {
                return this.onMessage(event);
            });
            this.socket.addEventListener("close", (event) => {
                return this.onClose(event);
            });
        }
        onClose(_event) {
            if (this.inst !== undefined) {
                this.globalObjects.forEach(obj => {
                    obj.dispose();
                });
                this.log(this.inst.runtimeStatsText());
                this.inst.dispose();
            }
            if (this.state === RPCServerState.ReceivePacketHeader) {
                this.log("Closing the server in clean state");
                this.log("Automatic reconnecting..");
                new RPCServer(this.url, this.key, this.getImports, this.logger, this.tensorCacheUrl, this.tensorCacheDevice, this.initProgressCallback, this.asyncOnServerLoad);
            }
            else {
                this.log("Closing the server, final state=" + this.state);
            }
        }
        onOpen(_event) {
            // Send the headers
            let bkey = StringToUint8Array("server:" + this.key);
            bkey = bkey.slice(0, bkey.length - 1);
            const intbuf = new Int32Array(1);
            intbuf[0] = RPC_MAGIC;
            this.socket.send(intbuf);
            intbuf[0] = bkey.length;
            this.socket.send(intbuf);
            this.socket.send(bkey);
            this.log("connected...");
            // request bytes: magic + keylen
            this.requestBytes(4 /* SizeOf.I32 */ + 4 /* SizeOf.I32 */);
            this.state = RPCServerState.InitHeader;
        }
        /** Handler for raw message. */
        onMessage(event) {
            const buffer = event.data;
            this.buffredBytes += buffer.byteLength;
            this.messageQueue.push(new Uint8Array(buffer));
            this.processEvents();
        }
        /** Process ready events. */
        processEvents() {
            while (this.buffredBytes >= this.pendingBytes && this.pendingBytes != 0) {
                this.onDataReady();
            }
        }
        /** State machine to handle each request */
        onDataReady() {
            switch (this.state) {
                case RPCServerState.InitHeader: {
                    this.handleInitHeader();
                    break;
                }
                case RPCServerState.InitHeaderKey: {
                    this.handleInitHeaderKey();
                    break;
                }
                case RPCServerState.ReceivePacketHeader: {
                    this.currPacketHeader = this.readFromBuffer(8 /* SizeOf.I64 */);
                    const reader = new ByteStreamReader(this.currPacketHeader);
                    this.currPacketLength = reader.readU64();
                    assert(this.pendingBytes === 0);
                    this.requestBytes(this.currPacketLength);
                    this.state = RPCServerState.ReceivePacketBody;
                    break;
                }
                case RPCServerState.ReceivePacketBody: {
                    const body = this.readFromBuffer(this.currPacketLength);
                    assert(this.pendingBytes === 0);
                    assert(this.currPacketHeader !== undefined);
                    this.onPacketReady(this.currPacketHeader, body);
                    break;
                }
                case RPCServerState.WaitForCallback: {
                    assert(this.pendingBytes === 0);
                    break;
                }
                default: {
                    throw new Error("Cannot handle state " + this.state);
                }
            }
        }
        onPacketReady(header, body) {
            if (this.inst === undefined) {
                // initialize server.
                const reader = new ByteStreamReader(body);
                reader.readU32();
                Uint8ArrayToString(reader.readByteArray());
                const nargs = reader.readU32();
                // nargs=0 means no session_constructor_args (LocalSession request).
                // WASM RPC requires ["rpc.WasmSession", wasm_binary]. Wait for proper init.
                if (nargs === 0) {
                    this.log("Received LocalSession init (nargs=0), waiting for WasmSession init...");
                    this.requestBytes(8 /* SizeOf.I64 */);
                    this.state = RPCServerState.ReceivePacketHeader;
                    return;
                }
                const args = [];
                for (let i = 0; i < nargs; ++i) {
                    const typeIndex = reader.readU32();
                    if (typeIndex === 8 /* TypeIndex.kTVMFFIRawStr */) {
                        const str = Uint8ArrayToString(reader.readByteArray());
                        args.push(str);
                    }
                    else if (typeIndex === 65 /* TypeIndex.kTVMFFIStr */) {
                        reader.readU32(); // skip duplicate type_index
                        const str = Uint8ArrayToString(reader.readByteArray());
                        args.push(str);
                    }
                    else if (typeIndex === 9 /* TypeIndex.kTVMFFIByteArrayPtr */) {
                        args.push(reader.readByteArray());
                    }
                    else if (typeIndex === 66 /* TypeIndex.kTVMFFIBytes */) {
                        reader.readU32(); // skip duplicate type_index
                        args.push(reader.readByteArray());
                    }
                    else {
                        throw new Error("cannot support type index " + typeIndex);
                    }
                }
                this.onInitServer(args, header, body);
            }
            else {
                assert(this.serverRecvData !== undefined);
                this.serverRecvData(header, body);
                this.requestBytes(8 /* SizeOf.I64 */);
                this.state = RPCServerState.ReceivePacketHeader;
            }
        }
        /** Event handler during server initialization. */
        onInitServer(args, header, body) {
            // start the server
            assert(args[0] === "rpc.WasmSession");
            assert(this.pendingBytes === 0);
            const asyncInitServer = () => __awaiter(this, void 0, void 0, function* () {
                assert(args[1] instanceof Uint8Array);
                const inst = yield instantiate(args[1].buffer, this.getImports(), this.logger);
                try {
                    const output = yield detectGPUDevice();
                    if (output !== undefined) {
                        const label = "WebGPU: " + output.adapterInfo.description;
                        this.log("Initialize GPU device: " + label);
                        inst.initWebGPU(output.device);
                    }
                    else {
                        this.log("Cannot find WebGPU device in the env");
                    }
                }
                catch (err) {
                    this.log("Cannnot initialize WebGPU, " + err.toString());
                }
                this.inst = inst;
                // begin scope to allow handling of objects
                this.inst.beginScope();
                if (this.initProgressCallback !== undefined) {
                    this.inst.registerInitProgressCallback(this.initProgressCallback);
                }
                if (this.tensorCacheUrl.length != 0) {
                    if (this.tensorCacheDevice === "cpu") {
                        yield this.inst.fetchTensorCache(this.tensorCacheUrl, this.inst.cpu());
                    }
                    else {
                        assert(this.tensorCacheDevice === "webgpu");
                        yield this.inst.fetchTensorCache(this.tensorCacheUrl, this.inst.webgpu());
                    }
                }
                assert(this.inst !== undefined);
                if (this.asyncOnServerLoad !== undefined) {
                    yield this.asyncOnServerLoad(this.inst);
                }
                const fcreate = this.inst.getGlobalFunc("rpc.CreateEventDrivenServer");
                const messageHandler = fcreate((cbytes) => {
                    assert(this.inst !== undefined);
                    if (this.socket.readyState === 1) {
                        // WebSocket will automatically close the socket
                        // if we burst send data that exceeds its internal buffer
                        // wait a bit before we send next one.
                        const sendDataWithCongestionControl = () => __awaiter(this, void 0, void 0, function* () {
                            const packetSize = 4 << 10;
                            const maxBufferAmount = 4 * packetSize;
                            const waitTimeMs = 20;
                            for (let offset = 0; offset < cbytes.length; offset += packetSize) {
                                const end = Math.min(offset + packetSize, cbytes.length);
                                while (this.socket.bufferedAmount >= maxBufferAmount) {
                                    yield new Promise((r) => setTimeout(r, waitTimeMs));
                                }
                                this.socket.send(cbytes.slice(offset, end));
                            }
                        });
                        // Chain up the pending send so that the async send is always in-order.
                        this.pendingSend = this.pendingSend.then(sendDataWithCongestionControl);
                        // Directly return since the data are "sent" from the caller's pov.
                        return this.inst.scalar(cbytes.length, "int32");
                    }
                    else {
                        return this.inst.scalar(0, "int32");
                    }
                }, this.name, this.key);
                // message handler should persist across RPC runs
                this.globalObjects.push(this.inst.detachFromCurrentScope(messageHandler));
                const writeFlag = this.inst.scalar(3, "int32");
                this.serverRecvData = (header, body) => {
                    if (messageHandler(header, writeFlag) === 0) {
                        this.socket.close();
                    }
                    if (messageHandler(body, writeFlag) === 0) {
                        this.socket.close();
                    }
                };
                // Forward the same init sequence to the wasm RPC.
                // The RPC will look for "rpc.wasmSession"
                // and we will redirect it to the correct local session.
                // register the callback to redirect the session to local.
                const flocal = this.inst.getGlobalFunc("wasm.LocalSession");
                const localSession = flocal();
                assert(localSession instanceof Module);
                this.inst.registerFunc("rpc.WasmSession", (_args) => {
                    return localSession;
                });
                messageHandler(header, writeFlag);
                messageHandler(body, writeFlag);
                this.log("Finish initializing the Wasm Server..");
                this.requestBytes(8 /* SizeOf.I64 */);
                this.state = RPCServerState.ReceivePacketHeader;
                // call process events in case there are bufferred data.
                this.processEvents();
                // recycle all values.
                this.inst.endScope();
            });
            this.state = RPCServerState.WaitForCallback;
            asyncInitServer();
        }
        log(msg) {
            this.logger(this.name + msg);
        }
        handleInitHeader() {
            const reader = new ByteStreamReader(this.readFromBuffer(4 /* SizeOf.I32 */ * 2));
            const magic = reader.readU32();
            if (magic === RPC_MAGIC + 1) {
                throw new Error("key: " + this.key + " has already been used in proxy");
            }
            else if (magic === RPC_MAGIC + 2) {
                throw new Error("RPCProxy do not have matching client key " + this.key);
            }
            assert(magic === RPC_MAGIC, this.url + " is not an RPC Proxy");
            this.remoteKeyLength = reader.readU32();
            assert(this.pendingBytes === 0);
            this.requestBytes(this.remoteKeyLength);
            this.state = RPCServerState.InitHeaderKey;
        }
        handleInitHeaderKey() {
            Uint8ArrayToString(this.readFromBuffer(this.remoteKeyLength));
            assert(this.pendingBytes === 0);
            this.requestBytes(8 /* SizeOf.I64 */);
            this.state = RPCServerState.ReceivePacketHeader;
        }
        checkLittleEndian() {
            const a = new ArrayBuffer(4);
            const b = new Uint8Array(a);
            const c = new Uint32Array(a);
            b[0] = 0x11;
            b[1] = 0x22;
            b[2] = 0x33;
            b[3] = 0x44;
            assert(c[0] === 0x44332211, "RPCServer little endian to work");
        }
        requestBytes(nbytes) {
            this.pendingBytes += nbytes;
        }
        readFromBuffer(nbytes) {
            const ret = new Uint8Array(nbytes);
            let ptr = 0;
            while (ptr < nbytes) {
                assert(this.messageQueue.length != 0);
                const nleft = nbytes - ptr;
                if (this.messageQueue[0].byteLength <= nleft) {
                    const buffer = this.messageQueue.shift();
                    ret.set(buffer, ptr);
                    ptr += buffer.byteLength;
                }
                else {
                    const buffer = this.messageQueue[0];
                    ret.set(buffer.slice(0, nleft), ptr);
                    this.messageQueue[0] = buffer.slice(nleft, buffer.byteLength);
                    ptr += nleft;
                }
            }
            this.buffredBytes -= nbytes;
            this.pendingBytes -= nbytes;
            return ret;
        }
    }

    exports.ArtifactCache = ArtifactCache;
    exports.ArtifactCrossOriginStorageCache = ArtifactCrossOriginStorageCache;
    exports.ArtifactIndexedDBCache = ArtifactIndexedDBCache;
    exports.ArtifactOPFSCache = ArtifactOPFSCache;
    exports.CacheState = CacheState;
    exports.DLDataType = DLDataType;
    exports.DLDevice = DLDevice;
    exports.Instance = Instance;
    exports.LRUCache = LRUCache;
    exports.LinearCongruentialGenerator = LinearCongruentialGenerator;
    exports.Module = Module;
    exports.RPCServer = RPCServer;
    exports.Scalar = Scalar;
    exports.TVMArray = TVMArray;
    exports.TVMObject = TVMObject;
    exports.Tensor = Tensor;
    exports.VirtualMachine = VirtualMachine;
    exports.assert = assert;
    exports.createArtifactCache = createArtifactCache;
    exports.createPolyfillWASI = createPolyfillWASI;
    exports.deleteTensorCache = deleteTensorCache;
    exports.detectGPUDevice = detectGPUDevice;
    exports.hasTensorInCache = hasTensorInCache;
    exports.instantiate = instantiate;
    exports.wasmPath = wasmPath;

}));
