# Model-level RLL suites are independent of the small field/unit tests.
if(VVM_TEST_RLL)
    vvm_add_model_test(test_jung2019_shared_model
        SCRIPT check_jung2019_model.py OUTPUT testing_output_jung2019
        TIMEOUT 300 LABELS rll)
    vvm_add_model_test(test_rll_mountain_shared_model
        SCRIPT check_rll_mountain.py OUTPUT rll_mountain_tests
        TIMEOUT 600 LABELS rll)
    if(VVM_TEST_BP5)
        vvm_add_model_test(test_rll_bp5_history
            SCRIPT check_rll_bp5.py OUTPUT rll_bp5_tests
            TIMEOUT 600 LABELS rll bp5)
        vvm_add_model_test(test_rll_bp5_restart
            SCRIPT check_rll_restart.py OUTPUT rll_bp5_restart_tests
            TIMEOUT 600 LABELS rll bp5 restart)
        _vvm_set_test_resources(test_rll_bp5_restart 2)
        vvm_add_model_test(test_wind_restart_modes
            SCRIPT check_wind_restart_modes.py OUTPUT wind_restart_modes_tests
            TIMEOUT 600 LABELS rll bp5 restart)
    endif()
endif()

if(VVM_TEST_RLL_PHYSICS)
    vvm_add_model_test(test_rll_turbulence_model
        SCRIPT check_rll_turbulence_model.py OUTPUT rll_turbulence_model_tests
        TIMEOUT 240 LABELS rll-physics)
    vvm_add_model_test(test_rll_surface_model
        SCRIPT check_rll_surface_model.py OUTPUT rll_surface_model_tests
        TIMEOUT 300 LABELS rll-physics)
    vvm_add_model_test(test_rll_turbulence_surface_model
        SCRIPT check_rll_turbulence_surface_model.py OUTPUT rll_turbulence_surface_model_tests
        TIMEOUT 300 LABELS rll-physics)
    vvm_add_model_test(test_rll_p3_initialization
        SCRIPT check_rll_p3_initialization.py OUTPUT rll_p3_initialization_tests
        TIMEOUT 300 LABELS rll-physics)
    vvm_add_model_test(test_rll_p3_uniform_model
        SCRIPT check_rll_p3_uniform_model.py OUTPUT rll_p3_uniform_model_tests
        TIMEOUT 360 LABELS rll-physics physics)
    vvm_add_model_test(test_rll_p3_model
        SCRIPT check_rll_p3_model.py OUTPUT rll_p3_model_tests
        TIMEOUT 420 LABELS rll-physics physics)
    vvm_add_model_test(test_rll_periodic_p3_model
        SCRIPT check_rll_p3_model.py OUTPUT rll_periodic_p3_model_tests
        TIMEOUT 420 LABELS rll-physics physics ARGS --periodic)
    vvm_add_model_test(test_rll_periodic_rest_model
        SCRIPT check_rll_p3_model.py OUTPUT rll_periodic_rest_model_tests
        TIMEOUT 420 LABELS rll-physics physics ARGS --periodic --rest)
endif()
