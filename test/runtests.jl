using RobustAdaptiveMetropolisSampler
using Distributions
using Statistics
using LinearAlgebra
using Random
using Test

@testset "RobustAdaptiveMetropolisSampler" begin

    Random.seed!(10)
    smplr = RAM([0.], 0.5)
    smplrd = RAM([0.], Diagonal([0.5]))
    smplrm = RAM([0.], fill(0.5,1,))
    smplrv = RAM([0.], [0.5])
    result = sample(p -> logpdf(Normal(3., 2.), p[1]),  smplr,        100_000, show_progress=false, output_log_probability_x=true)
    result2 = sample(p -> logpdf(Normal(3., 2.), p[1]), result.state, 100_000, show_progress=false, output_log_probability_x=true)

    @test length(result.chain) == 100_000
    @test result.acceptance_rate ≈ 0.234 atol = 0.01
    @test mean(result.chain[:,1]) ≈ 3. atol = 0.1
    @test std(result.chain[:,1]) ≈ 2. atol = 0.01
    @test result.log_probabilities_x[1] == logpdf(Normal(3., 2.), result.chain[1])
    @test result.log_probabilities_x[length(result.chain)] == logpdf(Normal(3., 2.), result.chain[length(result.chain)])

    @test length(result2.chain) == 100_000
    @test result2.acceptance_rate ≈ 0.234 atol = 0.01
    @test mean(result2.chain[:,1]) ≈ 3. atol = 0.1
    @test std(result2.chain[:,1]) ≈ 2. atol = 0.01
    @test result2.log_probabilities_x[1] == logpdf(Normal(3., 2.), result2.chain[1])
    @test result2.log_probabilities_x[length(result2.chain)] == logpdf(Normal(3., 2.), result2.chain[length(result.chain)])


    result = sample(p -> logpdf(Normal(3., 2.), p[1]), smplrd, 1_000, show_progress=false, output_log_probability_x=false)
    @test length(result.chain) == 1_000

    result = sample(p -> logpdf(Normal(3., 2.), p[1]), smplrv, 1_000, show_progress=false, output_log_probability_x=false)
    @test length(result.chain) == 1_000

    result = sample(p -> logpdf(Normal(3., 2.), p[1]), smplrm, 1_000, show_progress=false, output_log_probability_x=false)
    @test length(result.chain) == 1_000

end
