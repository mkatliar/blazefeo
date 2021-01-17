BENCH_BLAS = build/bin/bench-blas
BENCH_BLASFEO = build/bin/bench-blasfeo
BENCH_BLAZE = build/bin/bench-blaze
BENCH_EIGEN = build/bin/bench-eigen
BENCH_BLAZEFEO = build/bin/bench-blazefeo
BENCH_LIBXSMM = build/bin/bench-libxsmm
BENCHMARK_OPTIONS = --benchmark_repetitions=5 --benchmark_counters_tabular=true --benchmark_out_format=json --benchmark_report_aggregates_only=true
RUN_MATLAB = matlab -nodisplay -nosplash -nodesktop -r
BENCH_DATA = bench_result/data
BENCH_IMAGE = bench_result/image
PYTHON = python3

# run-benchmark-blaze-dynamic:
# 	$(BENCH_TMPC) --benchmark_filter="BM_gemm_blaze_dynamic<double>*" $(BENCHMARK_OPTIONS) \
# 		--benchmark_out=${BENCH_DATA}/blaze-dynamic.json

${BENCH_DATA}/loop-naive.json:
	$(BENCH_TMPC) --benchmark_filter="BM_gemm_loop_naive<double>*" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/loop-naive.json

#
# DGEMM/SGEMM bechmarks data
#
${BENCH_DATA}/dgemm-openblas.json: $(BENCH_BLAS)-openblas
	$(BENCH_BLAS)-openblas --benchmark_filter="BM_gemm<double>*" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/dgemm-openblas.json

${BENCH_DATA}/dgemm-mkl.json: $(BENCH_BLAS)-mkl_rt
	$(BENCH_BLAS)-mkl_rt --benchmark_filter="BM_gemm<double>*" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/dgemm-mkl.json

${BENCH_DATA}/dgemm-blasfeo-blas.json: $(BENCH_BLAS)-blasfeo
	$(BENCH_BLAS)-blasfeo --benchmark_filter="BM_gemm<double>*" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/dgemm-blasfeo-blas.json

${BENCH_DATA}/dgemm-blasfeo.json: $(BENCH_BLASFEO)
	$(BENCH_BLASFEO) --benchmark_filter="BM_gemm<double>" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/dgemm-blasfeo.json

${BENCH_DATA}/sgemm-blasfeo.json: $(BENCH_BLASFEO)
	$(BENCH_BLASFEO) --benchmark_filter="BM_gemm<float>" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/sgemm-blasfeo.json

${BENCH_DATA}/dgemm-blaze-dynamic.json: $(BENCH_BLAZE)
	$(BENCH_BLAZE) --benchmark_filter="BM_gemm_dynamic<double>*" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/dgemm-blaze-dynamic.json

${BENCH_DATA}/dgemm-blaze-static.json: $(BENCH_BLAZE)
	$(BENCH_BLAZE) --benchmark_filter="BM_gemm_static<double>*" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/dgemm-blaze-static.json

${BENCH_DATA}/dgemm-eigen-dynamic.json: $(BENCH_EIGEN)
	$(BENCH_EIGEN) --benchmark_filter="BM_gemm_dynamic<double>*" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/dgemm-eigen-dynamic.json

${BENCH_DATA}/dgemm-eigen-static.json: $(BENCH_EIGEN)
	$(BENCH_EIGEN) --benchmark_filter="BM_gemm_static<double>*" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/dgemm-eigen-static.json

${BENCH_DATA}/sgemm-blaze-static.json: $(BENCH_BLAZE)
	$(BENCH_BLAZE) --benchmark_filter="BM_gemm_static<float>*" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/sgemm-blaze-static.json

${BENCH_DATA}/dgemm-blazefeo-static-panel.json: $(BENCH_BLAZEFEO)
	$(BENCH_BLAZEFEO) --benchmark_filter="BM_gemm_static_panel<double, .+>" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/dgemm-blazefeo-static-panel.json

${BENCH_DATA}/dgemm-blazefeo-dynamic-panel.json: $(BENCH_BLAZEFEO)
	$(BENCH_BLAZEFEO) --benchmark_filter="BM_gemm_dynamic_panel<double>" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/dgemm-blazefeo-dynamic-panel.json

${BENCH_DATA}/dgemm-blazefeo-static-plain.json: $(BENCH_BLAZEFEO)
	$(BENCH_BLAZEFEO) --benchmark_filter="BM_gemm_static_plain<double, .+>" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/dgemm-blazefeo-static-plain.json

${BENCH_DATA}/dgemm-blazefeo-dynamic-plain.json: $(BENCH_BLAZEFEO)
	$(BENCH_BLAZEFEO) --benchmark_filter="BM_gemm_dynamic_plain<double>" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/dgemm-blazefeo-dynamic-plain.json

${BENCH_DATA}/sgemm-blazefeo-static-panel.json: $(BENCH_BLAZEFEO)
	$(BENCH_BLAZEFEO) --benchmark_filter="BM_gemm_static_panel<float, .+>" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/sgemm-blazefeo-static-panel.json

${BENCH_DATA}/sgemm-blazefeo-dynamic-panel.json: $(BENCH_BLAZEFEO)
	$(BENCH_BLAZEFEO) --benchmark_filter="BM_gemm_dynamic_panel<float>" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/sgemm-blazefeo-dynamic-panel.json

${BENCH_DATA}/dgemm-libxsmm.json: $(BENCH_LIBXSMM)
	$(BENCH_LIBXSMM) --benchmark_filter="BM_gemm_nt<double>" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/dgemm-libxsmm.json

${BENCH_DATA}/sgemm-libxsmm.json: $(BENCH_LIBXSMM)
	$(BENCH_LIBXSMM) --benchmark_filter="BM_gemm_nt<float>" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/sgemm-libxsmm.json

dgemm-benchmarks: \
	${BENCH_DATA}/dgemm-libxsmm.json \
	${BENCH_DATA}/dgemm-blasfeo.json \
	${BENCH_DATA}/dgemm-blaze-static.json \
	${BENCH_DATA}/dgemm-eigen-static.json \
	${BENCH_DATA}/dgemm-blazefeo-static-panel.json \
	${BENCH_DATA}/dgemm-blazefeo-dynamic-panel.json \
	${BENCH_DATA}/dgemm-blazefeo-static-plain.json \
	${BENCH_DATA}/dgemm-blazefeo-dynamic-plain.json
	

#
# DPOTRF benchmark data
#
${BENCH_DATA}/dpotrf-blasfeo.json: $(BENCH_BLASFEO)
	$(BENCH_BLASFEO) --benchmark_filter="BM_potrf<double>" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/dpotrf-blasfeo.json

${BENCH_DATA}/dpotrf-blas-mkl_rt.json: $(BENCH_BLAS)-mkl_rt
	$(BENCH_BLAS)-mkl_rt --benchmark_filter="BM_potrf<double>" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/dpotrf-blas-mkl_rt.json

${BENCH_DATA}/dpotrf-blazefeo-static.json: $(BENCH_BLAZEFEO)
	$(BENCH_BLAZEFEO) --benchmark_filter="BM_potrf_static_panel.+" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/dpotrf-blazefeo-static-panel.json

${BENCH_DATA}/dpotrf-blazefeo-dynamic.json: $(BENCH_BLAZEFEO)
	$(BENCH_BLAZEFEO) --benchmark_filter="BM_potrf_dynamic_panel.+" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/dpotrf-blazefeo-dynamic-panel.json


#
# SPOTRF benchmark data
#
		
${BENCH_DATA}/spotrf-blasfeo.json: $(BENCH_BLASFEO)
	$(BENCH_BLASFEO) --benchmark_filter="BM_potrf<float>" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/spotrf-blasfeo.json

${BENCH_DATA}/spotrf-blas-mkl_rt.json: $(BENCH_BLAS)-mkl_rt
	$(BENCH_BLAS)-mkl_rt --benchmark_filter="BM_potrf<float>" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/spotrf-blas-mkl_rt.json

${BENCH_DATA}/spotrf-blazefeo-static.json: $(BENCH_BLAZEFEO)
	$(BENCH_BLAZEFEO) --benchmark_filter="BM_potrf_static_panel<float, .+>" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/spotrf-blazefeo-static-panel.json

${BENCH_DATA}/spotrf-blazefeo-dynamic.json: $(BENCH_BLAZEFEO)
	$(BENCH_BLAZEFEO) --benchmark_filter="BM_potrf_dynamic_panel<float>/.+" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/spotrf-blazefeo-dynamic-panel.json


#
# DSYRK benchmark data
#
${BENCH_DATA}/dsyrk-blasfeo.json: $(BENCH_BLASFEO)
	$(BENCH_BLASFEO) --benchmark_filter="BM_syrk<double>" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/dsyrk-blasfeo.json

${BENCH_DATA}/dsyrk-blas-mkl_rt.json: $(BENCH_BLAS)-mkl_rt
	$(BENCH_BLAS)-mkl_rt --benchmark_filter="BM_syrk<double>" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/dsyrk-blas-mkl_rt.json

${BENCH_DATA}/dsyrk-blazefeo-static-plain.json: $(BENCH_BLAZEFEO)
	$(BENCH_BLAZEFEO) --benchmark_filter="BM_syrk_static_plain.+" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/dsyrk-blazefeo-static-plain.json

${BENCH_DATA}/dsyrk-blazefeo-dynamic-plain.json: $(BENCH_BLAZEFEO)
	$(BENCH_BLAZEFEO) --benchmark_filter="BM_syrk_dynamic_plain.+" $(BENCHMARK_OPTIONS) \
		--benchmark_out=${BENCH_DATA}/dsyrk-blazefeo-dynamic-plain.json


#
# Performance plots
#
${BENCH_IMAGE}/dgemm_performance.png: bench/analysis/dgemm_performance.py dgemm-benchmarks
	$(PYTHON) bench/analysis/dgemm_performance.py

${BENCH_IMAGE}/dgemm_performance_ratio.png: bench/analysis/dgemm_performance_ratio.py dgemm-benchmarks
	$(PYTHON) bench/analysis/dgemm_performance_ratio.py

${BENCH_IMAGE}/dgemm_blazefeo_vs_blasfeo.tex: bench/analysis/dgemm_performance_ratio.py \
	${BENCH_DATA}/dgemm-blazefeo-static-plain.json\
	${BENCH_DATA}/dgemm-blazefeo-dynamic-plain.json\
	${BENCH_DATA}/dgemm-libxsmm.json\
	${BENCH_DATA}/dgemm-blasfeo.json\
	${BENCH_DATA}/dgemm-blaze-static.json\
	${BENCH_DATA}/dgemm-eigen-static.json
	$(RUN_MATLAB) "make_figure_blazefeo_vs_blasfeo('dgemm'); quit"

${BENCH_IMAGE}/sgemm_blazefeo_vs_blasfeo.tex: make_figure_blazefeo_vs_blasfeo.m \
	${BENCH_DATA}/sgemm-blazefeo-static.json\
	${BENCH_DATA}/sgemm-blazefeo-dynamic.json\
	${BENCH_DATA}/sgemm-libxsmm.json\
	${BENCH_DATA}/sgemm-blasfeo.json\
	${BENCH_DATA}/sgemm-blaze-static.json
	$(RUN_MATLAB) "make_figure_blazefeo_vs_blasfeo('sgemm'); quit"

${BENCH_IMAGE}/dgemm_performance_ratio.tex: make_figure_performance_ratio.m \
	${BENCH_DATA}/dgemm-blazefeo-static-inline.json\
	${BENCH_DATA}/dgemm-blazefeo-dynamic-inline.json\
	${BENCH_DATA}/dgemm-blasfeo.json
	$(RUN_MATLAB) "make_figure_performance_ratio('dgemm'); quit"

${BENCH_IMAGE}/panel_format_effect.tex: make_figure_panel_format_effect.m \
	${BENCH_DATA}/dgemm-blazefeo-static-panel.json \
	${BENCH_DATA}/dgemm-blazefeo-static-plain.json \
	${BENCH_DATA}/dgemm-blaze-static.json \
	${BENCH_DATA}/dgemm-eigen-static.json \
	${BENCH_DATA}/dgemm-blasfeo.json
	$(RUN_MATLAB) "make_figure_panel_format_effect; quit"

${BENCH_IMAGE}/dpotrf_performance.tex: make_figure_performance.m \
	${BENCH_DATA}/dpotrf-blazefeo-static.json \
	${BENCH_DATA}/dpotrf-blasfeo.json \
	${BENCH_DATA}/dpotrf-blas-mkl_rt.json \
	${BENCH_DATA}/dpotrf-blazefeo-dynamic.json
	$(RUN_MATLAB) "make_figure_performance('dpotrf'); quit"
	
${BENCH_IMAGE}/spotrf_performance.tex: make_figure_performance.m \
	${BENCH_DATA}/spotrf-blazefeo-static.json \
	${BENCH_DATA}/spotrf-blasfeo.json \
	${BENCH_DATA}/spotrf-blas-mkl_rt.json \
	${BENCH_DATA}/spotrf-blazefeo-dynamic.json
	$(RUN_MATLAB) "make_figure_performance('spotrf'); quit"

${BENCH_IMAGE}/riccati_performance.tex ${BENCH_IMAGE}/riccati_performance.png: make_figure_riccati_performance.m \
	${BENCH_DATA}/riccati-tmpc-static-classical.json \
	${BENCH_DATA}/riccati-tmpc-dynamic-classical.json \
	${BENCH_DATA}/riccati-tmpc-static-factorized.json \
	${BENCH_DATA}/riccati-tmpc-dynamic-factorized.json
	$(RUN_MATLAB) "make_figure_riccati_performance(); quit"

${BENCH_IMAGE}/factorized_riccati_performance.tex ${BENCH_IMAGE}/factorized_riccati_performance.png: make_figure_factorized_riccati_performance.m \
	${BENCH_DATA}/riccati-hpipm-factorized.json \
	${BENCH_DATA}/riccati-tmpc-static-factorized.json \
	${BENCH_DATA}/riccati-tmpc-dynamic-factorized.json
	$(RUN_MATLAB) "make_figure_factorized_riccati_performance(); quit"

${BENCH_IMAGE}/mpc_software.pdf_tex: ${BENCH_IMAGE}/mpc_software.svg
	/usr/bin/inkscape --without-gui --file=${BENCH_IMAGE}/mpc_software.svg --export-pdf=${BENCH_IMAGE}/mpc_software.pdf --export-latex --export-area-drawing

${BENCH_IMAGE}/mpc_software.pdf: ${BENCH_IMAGE}/mpc_software.svg
	/usr/bin/inkscape --without-gui --file=${BENCH_IMAGE}/mpc_software.svg --export-pdf=${BENCH_IMAGE}/mpc_software.pdf --export-area-drawing