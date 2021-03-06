# Alternative GNU Make workspace makefile autogenerated by Premake

ifndef config
  config=debug
endif

ifndef verbose
  SILENT = @
endif

ifeq ($(config),debug)
  cpp_nn_config = debug

else ifeq ($(config),release)
  cpp_nn_config = release

else
  $(error "invalid configuration $(config)")
endif

PROJECTS := cpp_nn

.PHONY: all clean help $(PROJECTS) 

all: $(PROJECTS)

cpp_nn:
ifneq (,$(cpp_nn_config))
	@echo "==== Building cpp_nn ($(cpp_nn_config)) ===="
	@${MAKE} --no-print-directory -C build/cpp_nn -f Makefile config=$(cpp_nn_config)
endif

clean:
	@${MAKE} --no-print-directory -C build/cpp_nn -f Makefile clean

help:
	@echo "Usage: make [config=name] [target]"
	@echo ""
	@echo "CONFIGURATIONS:"
	@echo "  debug"
	@echo "  release"
	@echo ""
	@echo "TARGETS:"
	@echo "   all (default)"
	@echo "   clean"
	@echo "   cpp_nn"
	@echo ""
	@echo "For more information, see https://github.com/premake/premake-core/wiki"