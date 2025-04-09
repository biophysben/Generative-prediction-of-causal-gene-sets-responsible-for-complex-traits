#### Directory containing the input files for comparing with IBD
Files include:
1. all_webtwas.xlsx --- excel file of TWAS genes obtained from the TWAS database
2. gutjnl-2023-July-72-7-1271-inline-supplementary-material-1.xlsx --- excel file of the differentially expressed genes contained derived from the IBD dataset; this file is derived with the supplementary material of doi: [https://doi.org/10.1136/gutjnl-2021-326451].
3. mart_export.txt.gz --- gzipped csv file of the ENSEMBL IDs and their Entrez gene symbols, should be similar to the file on Dryad/Google Drive
4. README.md --- this file
5. twas_atlas_downloadgenes.txt --- genes downloaded from the TWAS Atlas
6. TWASAtlas_IBD_all.csv --- twas genes associated with IBD
7. uopt_IBD_forward_df.csv.gz --- csv file containing a DataFrame of the optimal $u_{opt}$ vectors for 2,500 point-to-point optimizations
8. webTWAS_IBD.xlsx --- IBD-related genes downloaded from the webTWAS database


Note: ../gene_perturbations_for_ben.gctx --- cmappy gctx file containing the transcriptional response to the gene perturbations; 

This file is available on Dryad/Google Drive, and is downloaded into the root-level directory. 

This is the $B$ matrix before being projected onto the causal eigengenes.
