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

import { Scikit1D, Scikit2D } from '../types'
import { tf } from '../shared/globals'
import { convertToTensor1D, convertToTensor2D } from '../utils';

type Tensor1D = tf.Tensor1D


export interface ShuffleSplitParams {
  nSplits?: number;
  testSize?: number;
  trainSize?: number;
  randomState?: number;
}

export abstract class BaseShuffleSplit {
  protected _nSplits: number
  protected _testSize?: number
  protected _trainSize?: number
  protected _randomState?: number
  protected _defaultTestSize = 0.1

  public constructor({
    nSplits = 10,
    testSize,
    trainSize,
    randomState
  }: ShuffleSplitParams = {}) {
    this._nSplits = nSplits
    this._testSize = testSize
    this._trainSize = trainSize
    this._randomState = randomState
  }

  protected abstract iterIndices(
    X: Scikit2D,
    y?: Scikit1D,
    groups?: Scikit1D
  ): void; // @todo add proper return type

  public getNSplits(): number {
    return this._nSplits
  }

  public split(
    X: Scikit2D,
    y?: Scikit1D,
    groups?: Scikit1D
  // ): IterableIterator<{ trainIndex: Tensor1D; testIndex: Tensor1D }> { @todo release
  ): void { // todo temp mock
    const features = convertToTensor2D(X)
    const labels = y ? convertToTensor1D(y) : undefined
    const groupLabels = groups ? convertToTensor1D(groups) : undefined

    this.iterIndices(features, labels, groupLabels);

    // @todo
    /*
    for train, test in self._iter_indices(X, y, groups):
            yield train, test
     */
  }
}
