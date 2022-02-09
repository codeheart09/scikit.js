export interface StratifiedShuffleSplitParams {
  nSplits: number;
  trainSize?: number;
  testSize?: number;
  randomState?: number;
}

// @todo any extension?
export class StratifiedShuffleSplit {
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

  getNSplits() {

  }

  split() {

  }
}
