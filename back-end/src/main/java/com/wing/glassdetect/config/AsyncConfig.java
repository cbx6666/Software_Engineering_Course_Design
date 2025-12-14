package com.wing.glassdetect.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;

@Configuration
@EnableAsync
public class AsyncConfig {

    @Bean("asyncExecutor")
    public ThreadPoolTaskExecutor asyncExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(10);   // 核心线程数
        executor.setMaxPoolSize(50);    // 最大线程数
        executor.setQueueCapacity(100); // 等待队列
        executor.setThreadNamePrefix("GlassAsync-");
        executor.initialize();
        return executor;
    }
}