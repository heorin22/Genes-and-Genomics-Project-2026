nextflow.enable.dsl=2


params.variants = "data/cftr_variants.csv"
params.fasta = "data/CFTR.fasta"
params.outdir = "results"


process SCORE_VARIANTS {

    tag "ESM-2 CFTR variant scoring"

    publishDir "${params.outdir}",
        mode: "copy",
        overwrite: true

    input:
    path variants
    path fasta

    output:
    path "variant_scores.csv",
        emit: scores

    script:
    """
    python ${projectDir}/scripts/score_variants.py \
        --variants ${variants} \
        --fasta ${fasta} \
        --output variant_scores.csv
    """
}


process ANALYSE_RESULTS {

    tag "Statistical analysis"

    publishDir "${params.outdir}",
        mode: "copy",
        overwrite: true

    input:
    path scores

    output:
    path "descriptive_statistics.csv"
    path "tukey_results.csv"

    script:
    """
    python ${projectDir}/scripts/analyse_results.py \
        --input ${scores} \
        --stats-output descriptive_statistics.csv \
        --tukey-output tukey_results.csv
    """
}


process PLOT_RESULTS {

    tag "Plot CFTR variant scores"

    publishDir "${params.outdir}",
        mode: "copy",
        overwrite: true

    input:
    path scores

    output:
    path "cftr_llr_boxplot.png"

    script:
    """
    python ${projectDir}/scripts/plot_results.py \
        --input ${scores} \
        --output cftr_llr_boxplot.png
    """
}


workflow {

    variants_ch = Channel.fromPath(
        params.variants,
        checkIfExists: true
    )

    fasta_ch = Channel.fromPath(
        params.fasta,
        checkIfExists: true
    )

    SCORE_VARIANTS(
        variants_ch,
        fasta_ch
    )

    scores = SCORE_VARIANTS.out.scores

    ANALYSE_RESULTS(scores)
    PLOT_RESULTS(scores)
}
