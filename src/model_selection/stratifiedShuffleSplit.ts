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
import {
  Scikit1D,
  Scikit2D
} from '../types';
import { validateShuffleSplit } from './trainTestSplit'
import { getLength } from '../utils'
import { tf } from '../shared/globals'


export class StratifiedShuffleSplit extends BaseShuffleSplit {
  protected async iterIndices(X: Scikit2D, y: Scikit1D, groups?: Scikit1D) { // @todo add return type
    const nSamples = getLength(X)

    const [nTrain, nTest] = validateShuffleSplit(
      nSamples,
      this._testSize,
      this._trainSize,
      this._defaultTestSize
    )

    const { values, indices } = tf.unique(y);

    const nClasses = values.shape[0];
    const classCounts = tf.bincount(indices, [], nClasses)

    const minClass = await classCounts.min().data();
    if (minClass[0] < 2) {
      throw Error('The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.')
    }

    if (nTrain < nClasses) {
      throw Error(`The trainSize = ${nTrain} should be greater or equal to the number of classes = ${nClasses}`)
    }

    if (nTest < nClasses) {
      throw Error(`The testSize = ${nTrain} should be greater or equal to the number of classes = ${nClasses}`)
    }

    console.log(indices);
    console.log(await classCounts.data());

    // const classIndices =

    // @todo continue here
    /*
      # Find the sorted list of instances for each class:
      # (np.unique above performs a sort, so code is O(n logn) already)
      class_indices = np.split(
          np.argsort(y_indices, kind="mergesort"), np.cumsum(class_counts)[:-1]
      )

      rng = check_random_state(self.random_state)

      for _ in range(self.n_splits):
          # if there are ties in the class-counts, we want
          # to make sure to break them anew in each iteration
          n_i = _approximate_mode(class_counts, n_train, rng)
          class_counts_remaining = class_counts - n_i
          t_i = _approximate_mode(class_counts_remaining, n_test, rng)

          train = []
          test = []

          for i in range(n_classes):
              permutation = rng.permutation(class_counts[i])
              perm_indices_class_i = class_indices[i].take(permutation, mode="clip")

              train.extend(perm_indices_class_i[: n_i[i]])
              test.extend(perm_indices_class_i[n_i[i] : n_i[i] + t_i[i]])

          train = rng.permutation(train)
          test = rng.permutation(test)

          yield train, test
     */
  }
}
