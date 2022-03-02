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

type Tensor1D = tf.Tensor1D


export interface ShuffleSplitParams {
  nSplits?: number;
  testSize?: number;
  trainSize?: number;
  randomState?: number;
}

export abstract class BaseShuffleSplit {
  nSplits: number
  testSize?: number
  trainSize?: number
  randomState?: number
  defaultTestSize = 0.1

  protected constructor({
    nSplits = 10,
    testSize,
    trainSize,
    randomState
  }: ShuffleSplitParams = {}) {
    // @todo Shouldn't the validation be performed here? Why only when I call split?
    this.nSplits = nSplits
    this.testSize = testSize
    this.trainSize = trainSize
    this.randomState = randomState
  }

  protected abstract iterIndices(
    X: Scikit2D,
    y?: Scikit1D,
    groups?: Scikit1D
  ): void; // @todo add proper return type

  public getNumSplits(): number {
    return this.nSplits
  }

  public split(
    X: Scikit2D,
    y?: Scikit1D,
    groups?: Scikit1D
  ): IterableIterator<{ trainIndex: Tensor1D; testIndex: Tensor1D }> {
    // @todo continue here
  }
}
