#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static const char MAGIC[] = "LIGHTNEST_WRAPPER_V1";
#define MAGIC_LEN (sizeof(MAGIC) - 1)
#define FOOTER_LEN (MAGIC_LEN + 16)
#define BUFFER_SIZE (16 * 1024 * 1024)

static int copy_range(int input, int output, off_t offset, uint64_t length) {
    unsigned char *buffer = malloc(BUFFER_SIZE);
    if (!buffer) return -1;
    if (lseek(input, offset, SEEK_SET) < 0) {
        free(buffer);
        return -1;
    }
    while (length > 0) {
        size_t request = length < BUFFER_SIZE ? (size_t)length : BUFFER_SIZE;
        ssize_t read_count = read(input, buffer, request);
        if (read_count <= 0) {
            free(buffer);
            return -1;
        }
        unsigned char *cursor = buffer;
        ssize_t remaining = read_count;
        while (remaining > 0) {
            ssize_t written = write(output, cursor, (size_t)remaining);
            if (written <= 0) {
                free(buffer);
                return -1;
            }
            cursor += written;
            remaining -= written;
        }
        length -= (uint64_t)read_count;
    }
    free(buffer);
    return 0;
}

static int make_temp(const char *directory, const char *pattern, char *path, size_t path_size, int mode) {
    if (snprintf(path, path_size, "%s/%s-XXXXXX", directory, pattern) >= (int)path_size) return -1;
    int fd = mkstemp(path);
    if (fd < 0) return -1;
    if (fchmod(fd, mode) < 0) {
        close(fd);
        unlink(path);
        return -1;
    }
    return fd;
}

int main(void) {
    char self_path[4096];
    ssize_t self_length = readlink("/proc/self/exe", self_path, sizeof(self_path) - 1);
    if (self_length <= 0 || (size_t)self_length >= sizeof(self_path)) return 111;
    self_path[self_length] = '\0';

    int input = open(self_path, O_RDONLY | O_CLOEXEC);
    if (input < 0) return 112;
    struct stat metadata;
    if (fstat(input, &metadata) < 0 || metadata.st_size < FOOTER_LEN) return 113;

    unsigned char footer[FOOTER_LEN];
    if (pread(input, footer, sizeof(footer), metadata.st_size - FOOTER_LEN) != (ssize_t)sizeof(footer)
            || memcmp(footer, MAGIC, MAGIC_LEN) != 0) return 114;
    uint64_t base_size = 0, model_size = 0;
    memcpy(&base_size, footer + MAGIC_LEN, sizeof(base_size));
    memcpy(&model_size, footer + MAGIC_LEN + sizeof(base_size), sizeof(model_size));
    uint64_t payload_start = (uint64_t)metadata.st_size - FOOTER_LEN - base_size - model_size;
    if (payload_start < 1 || payload_start > (uint64_t)metadata.st_size) return 115;

    char base_path[128], model_path[128];
    const char *temp_directory = getenv("TMPDIR");
    if (!temp_directory || !*temp_directory) temp_directory = "/tmp";
    int base_fd = make_temp(temp_directory, "lightnest-base", base_path, sizeof(base_path), 0700);
    int model_fd = make_temp(temp_directory, "lightnest-model", model_path, sizeof(model_path), 0600);
    if (base_fd < 0 || model_fd < 0) return 116;
    if (copy_range(input, base_fd, (off_t)payload_start, base_size) < 0
            || copy_range(input, model_fd, (off_t)(payload_start + base_size), model_size) < 0) {
        unlink(base_path);
        unlink(model_path);
        return 117;
    }
    close(input);
    close(base_fd);
    close(model_fd);

    if (setenv("LIGHTNEST_SELF_EXE", self_path, 1) < 0
            || setenv("LIGHTNEST_LANGUAGE_MODEL", model_path, 1) < 0
            || setenv("LIGHTNEST_BASE_EXE", base_path, 1) < 0) {
        unlink(base_path);
        unlink(model_path);
        return 118;
    }
    execl(base_path, base_path, (char *)NULL);
    unlink(base_path);
    unlink(model_path);
    return 119;
}
