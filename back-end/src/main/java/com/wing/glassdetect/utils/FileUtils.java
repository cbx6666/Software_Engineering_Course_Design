package com.wing.glassdetect.utils;

import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.*;
import java.util.UUID;

public class FileUtils {
    public static Path[] saveTempFile(MultipartFile[] imageFiles) throws IOException {
        Path tempDir = Paths.get(System.getProperty("user.dir"), "imageUploaded");
        if (!Files.exists(tempDir)) {
            Files.createDirectories(tempDir);
        }

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
}
