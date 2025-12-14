package com.wing.glassdetect.utils;

import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Comparator;
import java.util.UUID;

public class FileUtils {
    public static Path[] saveTempFile(MultipartFile[] imageFiles) throws IOException {
        // 每个请求一个独立子目录
        Path tempDir = Paths.get(System.getProperty("user.dir"), "imageUploaded", UUID.randomUUID().toString());
        Files.createDirectories(tempDir);

        Path[] paths = new Path[imageFiles.length];
        for (int i = 0; i < imageFiles.length; i++) {
            MultipartFile file = imageFiles[i];
            Path tempFile = tempDir.resolve("upload-" + System.currentTimeMillis() + "-" + UUID.randomUUID() + ".jpg");
            Files.copy(file.getInputStream(), tempFile, StandardCopyOption.REPLACE_EXISTING);
            paths[i] = tempFile;
        }
        return paths;
    }

    public static void deleteTempFile(Path tempFile) {
        if (tempFile != null) {
            try {
                Files.deleteIfExists(tempFile);
            } catch (IOException ignored) {}
        }
    }

    public static void deleteTempDir(Path tempDir) {
        if (tempDir != null && Files.exists(tempDir)) {
            try (var stream = Files.walk(tempDir)) { // try-with-resources
                stream.sorted(Comparator.reverseOrder())
                        .forEach(path -> {
                            try {
                                Files.deleteIfExists(path);
                            } catch (IOException ignored) {}
                        });
            } catch (IOException ignored) {}
        }
    }
}
