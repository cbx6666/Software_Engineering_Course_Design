package com.wing.glassdetect.controller;

import com.wing.glassdetect.dto.auth.LoginRequest;
import com.wing.glassdetect.dto.auth.LoginResponse;
import com.wing.glassdetect.dto.auth.RegisterRequest;
import com.wing.glassdetect.service.AuthService;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/auth")
public class AuthController {

    private final AuthService authService;

    public AuthController(AuthService authService) {
        this.authService = authService;
    }

    @PostMapping("/login")
    public LoginResponse login(@Valid @RequestBody LoginRequest req) {
        return authService.login(req);
    }

    @PostMapping("/register")
    public LoginResponse register(@Valid @RequestBody RegisterRequest req) {
        return authService.register(req);
    }
}
