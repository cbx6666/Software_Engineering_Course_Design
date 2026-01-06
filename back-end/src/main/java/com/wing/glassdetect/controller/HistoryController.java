package com.wing.glassdetect.controller;

import com.wing.glassdetect.model.History;
import com.wing.glassdetect.service.HistoryService;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/history")
public class HistoryController {

    private final HistoryService historyService;

    public HistoryController(HistoryService historyService) {
        this.historyService = historyService;
    }

    @GetMapping
    public List<History> getHistory(@RequestParam("userId") Long userId) {
        return historyService.getHistoryByUserId(userId);
    }

    @GetMapping("/{id}")
    public History getHistoryItemById(@PathVariable("id") Long id) {
        return historyService.getHistoryById(id);
    }
}

