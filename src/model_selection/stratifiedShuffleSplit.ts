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
  protected iterIndices(X: Scikit2D, y: Scikit1D, groups?: Scikit1D) {
    const nSamples = getLength(X)

    const [nTrain, nTest] = validateShuffleSplit(
      this._nSplits,
      this._testSize,
      this._trainSize,
      this._defaultTestSize
    )

    const { classes, y_indices } = tf.unique(y);
    const nClasses = classes.shape()[0];

  }
  // @todo continue here
  /*
    class_counts = np.bincount(y_indices)
    if np.min(class_counts) < 2:
        raise ValueError(
            "The least populated class in y has only 1"
            " member, which is too few. The minimum"
            " number of groups for any class cannot"
            " be less than 2."
        )

    if n_train < n_classes:
        raise ValueError(
            "The train_size = %d should be greater or "
            "equal to the number of classes = %d" % (n_train, n_classes)
        )
    if n_test < n_classes:
        raise ValueError(
            "The test_size = %d should be greater or "
            "equal to the number of classes = %d" % (n_test, n_classes)
        )

    # Find the sorted list of instances for each class:
    # (np.unique above performs a sort, so code is O(n logn) already)
    class_indices = np.split(
        np.argsort(y_indices, kind="mergesort"), np.cumsum(class_counts)[:-1]
    )

    rng = check_random_state(self.random_state)
   */
}
