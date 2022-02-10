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

import { CrossValidator } from './crossValidator';
import { Scikit1D, Scikit2D } from '../types';
import { assert } from '../typesUtils'

export interface StratifiedShuffleSplitParams {
  nSplits?: number;
  testSize?: number;
  trainSize?: number;
  randomState?: number;
}

export class StratifiedShuffleSplit implements CrossValidator {
  nSplits: number
  testSize?: number
  trainSize?: number
  randomState?: number

  constructor({
    nSplits = 10,
    testSize,
    trainSize,
    randomState
  }: StratifiedShuffleSplitParams = {}) {
    // Parse & validate: nSplits
    nSplits = Number(nSplits)
    assert(
      Number.isInteger(nSplits) && nSplits > 1,
      'new StratifiedShuffleSplit({nSplits}): nSplits must be an int greater than 1.'
    )

    // Parse & validate: testSize
    if (testSize === undefined && trainSize === undefined) {
      testSize = 0.1
    }

    // @todo validate before storing
    this.nSplits = nSplits
    this.trainSize = trainSize
    this.testSize = testSize
    this.randomState = randomState
  }

  getNumSplits(X: Scikit2D, y?: Scikit1D, groups?: Scikit1D): number {

  }

  split(
    X: Scikit2D,
    y?: Scikit1D,
    groups?: Scikit1D
  ): IterableIterator<{ trainIndex: Tensor1D; testIndex: Tensor1D }> {

  }
}
