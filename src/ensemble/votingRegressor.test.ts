import { makeVotingRegressor, VotingRegressor } from './votingRegressor'
import { DummyRegressor } from '../dummy/dummyRegressor'
import { LinearRegression } from '../linear_model/linearRegression'

describe('VotingRegressor', function () {
  it('Use VotingRegressor on simple example ', async function () {
    const X = [
      [1, 2],
      [2, 1],
      [2, 2],
      [3, 1]
    ]
    const y = [3, 3, 4, 4]
    const voter = new VotingRegressor({
      estimators: [
        ['dt', new DummyRegressor()],
        ['lr', new LinearRegression({ fitIntercept: true })]
      ]
    })

    await voter.fit(X, y)
    expect(voter.score(X, y) > 0).toBe(true)
  }, 10000)
  it('Use VotingRegressor on simple example ', async function () {
    const X = [
      [1, 2],
      [2, 1],
      [2, 2],
      [3, 1]
    ]
    const y = [3, 3, 4, 4]
    const voter = makeVotingRegressor(
      new DummyRegressor(),
      new LinearRegression({ fitIntercept: true })
    )

    await voter.fit(X, y)
    expect(voter.score(X, y) > 0).toBe(true)
  }, 10000)
})
