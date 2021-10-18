/**
*  @license
* Copyright 2021, JsData. All rights reserved.
*
* This source code is licensed under the MIT license found in the
* LICENSE file in the root directory of this source tree.

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* ==========================================================================
*/

import { Tensor, Tensor1D, Tensor2D } from '@tensorflow/tfjs-core'
import { DataFrame, Series } from 'danfojs-node'
import { DataType, TensorLike } from '@tensorflow/tfjs-node'

// The Types that Scikit uses
export type TypedArray = Float32Array | Int32Array | Uint8Array
export type ScikitLike1D = TypedArray | number[] | boolean[] | string[]
export type ScikitLike2D = TypedArray[] | number[][] | boolean[][] | string[][]
export type Scikit1D = ScikitLike1D | Tensor1D | Series
export type Scikit2D = ScikitLike2D | Tensor2D | DataFrame
export type ScikitVecOrMatrix = Scikit1D | Scikit2D

export type ArrayType1D = Array<
  number | string | boolean | (number | string | boolean)
>

export type ArrayType2D = Array<
  number[] | string[] | boolean[] | (number | string | boolean)[]
>

// Boolean functions to check all types defined above
export function isString(value: {}): value is string {
  return typeof value === 'string' || value instanceof String
}

export function isBoolean(value: {}): boolean {
  return typeof value === 'boolean'
}

export function isNumber(value: {}): boolean {
  return typeof value === 'number'
}

export function assert(expr: boolean, msg: () => string) {
  if (!expr) {
    throw new Error(typeof msg === 'string' ? msg : msg())
  }
}
export function inferShape(val: TensorLike, dtype?: DataType): number[] {
  let firstElem: typeof val = val

  if (isTypedArray(val)) {
    return dtype === 'string' ? [] : [val.length]
  }
  if (!Array.isArray(val)) {
    return [] // Scalar.
  }
  const shape: number[] = []

  while (
    Array.isArray(firstElem) ||
    (isTypedArray(firstElem) && dtype !== 'string')
  ) {
    shape.push(firstElem.length)
    firstElem = firstElem[0]
  }
  if (Array.isArray(val)) {
    deepAssertShapeConsistency(val, shape, [])
  }

  return shape
}

function deepAssertShapeConsistency(
  val: TensorLike,
  shape: number[],
  indices: number[]
) {
  indices = indices || []
  if (!Array.isArray(val) && !isTypedArray(val)) {
    assert(
      shape.length === 0,
      () =>
        `Element arr[${indices.join('][')}] is a primitive, ` +
        `but should be an array/TypedArray of ${shape[0]} elements`
    )
    return
  }
  assert(
    shape.length > 0,
    () =>
      `Element arr[${indices.join('][')}] should be a primitive, ` +
      `but is an array of ${val.length} elements`
  )
  assert(
    val.length === shape[0],
    () =>
      `Element arr[${indices.join('][')}] should have ${shape[0]} ` +
      `elements, but has ${val.length} elements`
  )
  const subShape = shape.slice(1)
  for (let i = 0; i < val.length; ++i) {
    deepAssertShapeConsistency(val[i], subShape, indices.concat(i))
  }
}

export function inferDtype(values: TensorLike): DataType | null {
  if (Array.isArray(values)) {
    return inferDtype(values[0])
  }
  if (values instanceof Float32Array) {
    return 'float32'
  } else if (values instanceof Int32Array || values instanceof Uint8Array) {
    return 'int32'
  } else if (isNumber(values)) {
    return 'float32'
  } else if (isString(values)) {
    return 'string'
  } else if (isBoolean(values)) {
    return 'bool'
  }
  // Failed inference
  return null
}
export function isTypedArray(a: {}): a is
  | Float32Array
  | Int32Array
  | Uint8Array {
  return (
    a instanceof Float32Array ||
    a instanceof Int32Array ||
    a instanceof Uint8Array
  )
}

export function isScikitLike1D(arr: any): arr is ScikitLike1D {
  const shape = inferShape(arr)
  const dtype = inferDtype(arr)
  return shape.length === 1 && dtype !== null
}
export function isScikitLike2D(arr: any): arr is ScikitLike2D {
  const shape = inferShape(arr)
  const dtype = inferDtype(arr)
  return shape.length === 2 && dtype !== null
}

export function isScikit1D(arr: any): arr is Scikit1D {
  if (arr instanceof Series) {
    return true
  }
  if (arr instanceof Tensor) {
    return arr.rank === 1
  }
  return isScikitLike1D(arr)
}

export function isScikit2D(arr: any): arr is Scikit2D {
  if (arr instanceof DataFrame) {
    return true
  }
  if (arr instanceof Tensor) {
    return arr.rank === 2
  }
  return isScikitLike2D(arr)
}

export function isScikitVecOrMatrix(arr: any): arr is ScikitVecOrMatrix {
  return isScikit1D(arr) || isScikit2D(arr)
}
