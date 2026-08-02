# Generate the two configs a rank-invariance test needs (1 rank and N ranks).
# A CTest command is a single process, so this wrapper issues both calls.
foreach(pair "${CFG1};${ODIR1}" "${CFGN};${ODIRN}")
    list(GET pair 0 cfg)
    list(GET pair 1 odir)
    execute_process(
        COMMAND ${PY} ${SCRIPT} --case ${CASE} --source-dir ${SRC}
                --out-config ${cfg} --out-dir ${odir} --seconds ${SECONDS}
        RESULT_VARIABLE rc)
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR "make_test_config.py failed for ${CASE} (${cfg})")
    endif()
endforeach()
