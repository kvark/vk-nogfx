CC ?= gcc
CFLAGS ?= -O2 -fPIC -Wall -Wextra -Wno-unused-parameter
LDFLAGS ?= -shared -ldl -pthread
LIB = libVkLayer_NOGFX_compute_first.so
JSON = VkLayer_NOGFX_compute_first.json
IMPLICIT_DIR = /usr/share/vulkan/implicit_layer.d

$(LIB): layer.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

install: $(LIB)
	install -d $(IMPLICIT_DIR)
	install -m 644 $(JSON) $(IMPLICIT_DIR)/$(JSON)

uninstall:
	rm -f $(IMPLICIT_DIR)/$(JSON)

clean:
	rm -f $(LIB)
