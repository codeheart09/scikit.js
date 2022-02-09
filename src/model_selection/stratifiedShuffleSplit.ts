import { CrossValidator } from './crossValidator';
import { Scikit1D, Scikit2D } from '../types';

export interface StratifiedShuffleSplitParams {
  nSplits: number;
  trainSize?: number;
  testSize?: number;
  randomState?: number;
}

export class StratifiedShuffleSplit implements CrossValidator {
  nSplits
  trainSize
  testSize
  randomState

  constructor({
    nSplits,
    trainSize,
    testSize,
    randomState
  }: StratifiedShuffleSplitParams = { nSplits: 10 }) {
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
