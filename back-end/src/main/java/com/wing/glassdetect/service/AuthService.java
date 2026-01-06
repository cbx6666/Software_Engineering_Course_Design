package com.wing.glassdetect.service;

import com.wing.glassdetect.dto.auth.LoginRequest;
import com.wing.glassdetect.dto.auth.LoginResponse;
import com.wing.glassdetect.dto.auth.RegisterRequest;
import com.wing.glassdetect.mapper.UserMapper;
import com.wing.glassdetect.model.User;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;

@Service
public class AuthService {

    private final UserMapper userMapper;
    private final PasswordEncoder passwordEncoder = new BCryptPasswordEncoder();

    public AuthService(UserMapper userMapper) {
        this.userMapper = userMapper;
    }

    public LoginResponse register(RegisterRequest req) {
        String email = normalizeEmail(req.getEmail());
        if (email.isBlank()) {
            throw new IllegalArgumentException("email 不能为空");
        }

        Long cnt = userMapper.selectCount(new LambdaQueryWrapper<User>().eq(User::getEmail, email));
        if (cnt != null && cnt > 0) {
            throw new IllegalArgumentException("该邮箱已注册");
        }

        User user = new User();
        user.setEmail(email);
        user.setPasswordHash(passwordEncoder.encode(req.getPassword()));
        userMapper.insert(user);

        if (user.getId() == null) {
            throw new IllegalStateException("创建用户失败：未获取到 id");
        }

        return new LoginResponse(new LoginResponse.User(String.valueOf(user.getId()), user.getEmail()));
    }

    public LoginResponse login(LoginRequest req) {
        String email = normalizeEmail(req.getEmail());
        if (email.isBlank()) {
            throw new IllegalArgumentException("email 不能为空");
        }

        User user = userMapper.selectOne(new LambdaQueryWrapper<User>().eq(User::getEmail, email));
        if (user == null || user.getPasswordHash() == null || user.getPasswordHash().isBlank()) {
            throw new IllegalArgumentException("用户不存在或未设置密码");
        }

        if (!passwordEncoder.matches(req.getPassword(), user.getPasswordHash())) {
            throw new IllegalArgumentException("邮箱或密码错误");
        }

        return new LoginResponse(new LoginResponse.User(String.valueOf(user.getId()), user.getEmail()));
    }

    private static String normalizeEmail(String raw) {
        return raw == null ? "" : raw.trim().toLowerCase();
    }

}
