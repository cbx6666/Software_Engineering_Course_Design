package com.wing.glassdetect.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.wing.glassdetect.model.DetectionResult;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.*;
import java.nio.file.*;
import java.util.stream.Collectors;

@Service
public class GlassCrackService {

    public DetectionResult detect(MultipartFile imageFile) {
        Path tempFile = null;

        try {
            // 保存图片到临时目录(放在项目根目录下 /imageUploaded)
            Path tempDir = Paths.get(System.getProperty("user.dir"), "imageUploaded");
            if (!Files.exists(tempDir)) {
                Files.createDirectories(tempDir);
            }

            tempFile = tempDir.resolve("upload-" + System.currentTimeMillis() + ".jpg");
            Files.copy(imageFile.getInputStream(), tempFile, StandardCopyOption.REPLACE_EXISTING);

            // 构造 Python 命令
            String pythonExe = "python3";
            // TODO: 待修改算法路径
            String scriptPath = Paths.get(
                    System.getProperty("user.dir"),
                    "../glass-algorithm/algorithm/glass_crack_detect.py"
            ).toAbsolutePath().normalize().toString();

            ProcessBuilder builder = new ProcessBuilder(
                    pythonExe,
                    scriptPath,
                    tempFile.toString()
            );
            builder.redirectErrorStream(true); // 合并标准输出和错误输出

            // 启动算法进程
            Process process = builder.start();

            // 读取算法输出
            String output;
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                output = reader.lines().collect(Collectors.joining());
            }

            int exitCode = process.waitFor();
            if (exitCode != 0) {
                return new DetectionResult("error", "裂纹检测失败", "Python 进程返回错误码：" + exitCode, null);
            }

            // 尝试解析 JSON 输出
            ObjectMapper mapper = new ObjectMapper();
            return mapper.readValue(output, DetectionResult.class);

        } catch (Exception e) {
            return new DetectionResult(
                    "error",
                    "检测失败",
                    "算法运行异常：" + e.getMessage(),
                    null
            );
        } finally {
            // 删除临时文件
            if (tempFile != null) {
                try {
                    Files.deleteIfExists(tempFile);
                } catch (IOException ignored) {}
            }
        }
    }
}
