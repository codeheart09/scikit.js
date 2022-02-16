/**
 *  @license
 * Copyright 2022, JsData. All rights reserved.
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

import { BaseShuffleSplit } from './baseShuffleSplit'
import { Scikit1D, Scikit2D } from '../types'
import { validateShuffleSplit } from './trainTestSplit'
import { assert } from '../typesUtils';
import { getLength } from '../utils';

// @todo pass to types
type CheckArrayOptions = {
  acceptSparse?: boolean
  acceptLargeSparse?: boolean
  dtype?: string
  order?: 'F' | 'C'
  copy?: boolean
  forceAllFinite?: boolean
  ensure2D?: boolean
  allowNd?: boolean
  ensureMinSamples?: number
  ensureMinFeatures?: number
  estimator?: string //@todo py: str or estimator instance (lookup estimator instance)
  inputName?: string
}

// @todo pass to utils
function checkArray(
  array,
  {
    dtype = 'number', // @todo should this be DataType type?
    forceAllFinite = true,
    ensure2D = true,
    ensureMinSamples = 1,
    ensureMinFeatures = 1
  }: CheckArrayOptions = {}
) {
  /*
    @todo needed?
    # store reference to original array to check if copy is needed when
    # function returns
    array_orig = array
   */

  // Store whether originally we wanted numeric dtype
  const dtypeNumeric = dtype === 'numeric'
  const dtypeOrig = array.dType || null // @todo this covers tensor, but how to do that on danfo?

  /*
    @todo how to make this check with tf / danfo?
    if not hasattr(dtype_orig, "kind"):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None
   */

  /*
    # check if the object contains several dtypes (typically a pandas
    # DataFrame), and store them. If not, store None.
    dtypes_orig = None
    pandas_requires_conversion = False
    if hasattr(array, "dtypes") and hasattr(array.dtypes, "__array__"):
        # throw warning if columns are sparse. If all columns are sparse, then
        # array.sparse exists and sparsity will be preserved (later).
        with suppress(ImportError):
            from pandas.api.types import is_sparse

            if not hasattr(array, "sparse") and array.dtypes.apply(is_sparse).any():
                warnings.warn(
                    "pandas.DataFrame with sparse columns found."
                    "It will be converted to a dense numpy array."
                )

        dtypes_orig = []
        for dtype_iter in array.dtypes:
            if dtype_iter.kind == "b":
                # pandas boolean dtype __array__ interface coerces bools to objects
                dtype_iter = np.dtype(object)
            elif _pandas_dtype_needs_early_conversion(dtype_iter):
                pandas_requires_conversion = True

            dtypes_orig.append(dtype_iter)

        if all(isinstance(dtype_iter, np.dtype) for dtype_iter in dtypes_orig):
            dtype_orig = np.result_type(*dtypes_orig)
   */
}

export interface StratifiedShuffleSplitParams {
  nSplits?: number;
  testSize?: number;
  trainSize?: number;
  randomState?: number;
}

export class StratifiedShuffleSplit implements BaseShuffleSplit {
  nSplits: number
  testSize?: number
  trainSize?: number
  randomState?: number
  defaultTestSize = 0.1

  constructor({
    nSplits = 10,
    testSize,
    trainSize,
    randomState
  }: StratifiedShuffleSplitParams = {}) {
    // @todo Shouldn't the validation be performed here? Why only when I call split?
    this.nSplits = nSplits
    this.testSize = testSize
    this.trainSize = trainSize
    this.randomState = randomState
  }

  getNumSplits(X: Scikit2D, y?: Scikit1D, groups?: Scikit1D): number {

  }

  // @todo How should I mark the method as private? Comment? Underscore?
  iterIndices(X: Scikit2D, y?: Scikit1D, groups?: Scikit1D) {
    const nSamples = getLength(X)
    y = checkArray() // @todo to be implemented

    const [nTrain, nTest] = validateShuffleSplit(
      this.nSplits,
      this.testSize,
      this.trainSize,
      this.defaultTestSize
    )

    // @todo would this validation be valid? (ba-dum-ts!) ;)
    assert(
      this.randomState === undefined || typeof this.randomState === 'number',
      `Invalid value for randomState: ${this.randomState}. Must be number or undefined`
    )

    if (typeof this.randomState === 'number') {
      assert(
        Number.isInteger(this.randomState) && this.randomState > 0,
        'If parameter randomState is provided, it must be an integer > 0'
      )
    }
  }

  split(
    X: Scikit2D,
    y?: Scikit1D,
    groups?: Scikit1D
  ): IterableIterator<{ trainIndex: Tensor1D; testIndex: Tensor1D }> {
    y = checkArray() // @todo why here and in iterIndices again?

  }
}
