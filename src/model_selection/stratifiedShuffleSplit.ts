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

  }
}
