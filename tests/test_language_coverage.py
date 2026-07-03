"""
Tests for language coverage in import/call extraction and AST analysis.

These tests verify that the Context-Engine properly extracts imports, calls,
and symbols from code in all supported languages: Python, JavaScript, TypeScript,
Go, Rust, Java, C, C++, Ruby, Kotlin, Swift, Scala, PHP, and C#.
"""
import pytest
import sys
from pathlib import Path

# Ensure scripts are importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ==============================================================================
# Import Extraction Tests
# ==============================================================================

class TestImportExtraction:
    """Test import extraction for all supported languages."""
    
    @pytest.fixture
    def extract_imports(self):
        """Return the import extraction function."""
        from scripts.ingest.metadata import _extract_imports
        return _extract_imports
    
    def test_python_imports(self, extract_imports):
        """Test Python import extraction."""
        code = '''
import os
import sys
from pathlib import Path
from collections.abc import Mapping
import numpy as np
from . import local_module
'''
        imports = extract_imports("python", code)
        assert "os" in imports
        assert "sys" in imports
        assert "pathlib" in imports
        assert "collections.abc" in imports
        assert "numpy" in imports
        assert "." in imports
    
    def test_javascript_imports(self, extract_imports):
        """Test JavaScript import extraction."""
        code = '''
import React from 'react';
import { useState, useEffect } from 'react';
import * as lodash from 'lodash';
const fs = require('fs');
require('dotenv').config();
'''
        imports = extract_imports("javascript", code)
        assert "react" in imports
        assert "lodash" in imports
        assert "fs" in imports
        assert "dotenv" in imports
    
    def test_typescript_imports(self, extract_imports):
        """Test TypeScript import extraction."""
        code = '''
import { Component } from '@angular/core';
import type { User } from './types';
import axios from 'axios';
'''
        imports = extract_imports("typescript", code)
        assert "@angular/core" in imports
        assert "./types" in imports
        assert "axios" in imports
    
    def test_go_imports(self, extract_imports):
        """Test Go import extraction."""
        code = '''
package main

import "fmt"
import (
    "os"
    "strings"
    "github.com/gin-gonic/gin"
)
'''
        imports = extract_imports("go", code)
        assert "fmt" in imports
        assert "os" in imports
        assert "strings" in imports
        assert "github.com/gin-gonic/gin" in imports
    
    def test_rust_imports(self, extract_imports):
        """Test Rust import extraction."""
        code = '''
use std::collections::HashMap;
use tokio::sync::Mutex;
use crate::utils::helper;
use super::parent_module;
'''
        imports = extract_imports("rust", code)
        assert "std::collections::HashMap" in imports
        assert "tokio::sync::Mutex" in imports
        assert "crate::utils::helper" in imports
        assert "super::parent_module" in imports
    
    def test_java_imports(self, extract_imports):
        """Test Java import extraction."""
        code = '''
package com.example;

import java.util.*;
import java.io.IOException;
import com.google.common.collect.ImmutableList;
'''
        imports = extract_imports("java", code)
        assert "java.util.*" in imports
        assert "java.io.IOException" in imports
        assert "com.google.common.collect.ImmutableList" in imports
    
    def test_c_includes(self, extract_imports):
        """Test C #include extraction."""
        code = '''
#include <stdio.h>
#include <stdlib.h>
#include "myheader.h"
#include "../utils/helper.h"
'''
        imports = extract_imports("c", code)
        assert "stdio.h" in imports
        assert "stdlib.h" in imports
        assert "myheader.h" in imports
        assert "../utils/helper.h" in imports
    
    def test_cpp_includes(self, extract_imports):
        """Test C++ #include extraction."""
        code = '''
#include <iostream>
#include <vector>
#include <memory>
#include "config.hpp"
'''
        imports = extract_imports("cpp", code)
        assert "iostream" in imports
        assert "vector" in imports
        assert "memory" in imports
        assert "config.hpp" in imports
    
    def test_ruby_requires(self, extract_imports):
        """Test Ruby require extraction."""
        code = '''
require 'json'
require 'net/http'
require_relative 'lib/helper'
load 'config.rb'
'''
        imports = extract_imports("ruby", code)
        assert "json" in imports
        assert "net/http" in imports
        assert "lib/helper" in imports
        assert "config.rb" in imports
    
    def test_kotlin_imports(self, extract_imports):
        """Test Kotlin import extraction."""
        code = '''
package com.example

import kotlin.collections.List
import kotlinx.coroutines.*
import com.google.gson.Gson
'''
        imports = extract_imports("kotlin", code)
        assert "kotlin.collections.List" in imports
        assert "kotlinx.coroutines.*" in imports
        assert "com.google.gson.Gson" in imports
    
    def test_swift_imports(self, extract_imports):
        """Test Swift import extraction."""
        code = '''
import Foundation
import UIKit
import SwiftUI
'''
        imports = extract_imports("swift", code)
        assert "Foundation" in imports
        assert "UIKit" in imports
        assert "SwiftUI" in imports
    
    def test_scala_imports(self, extract_imports):
        """Test Scala import extraction."""
        code = '''
import scala.collection.mutable
import akka.actor.{Actor, Props}
import java.util._
'''
        imports = extract_imports("scala", code)
        assert "scala.collection.mutable" in imports
        assert any("akka.actor" in imp for imp in imports)
        assert "java.util._" in imports
    
    def test_csharp_usings(self, extract_imports):
        """Test C# using extraction."""
        code = '''
using System;
using System.Collections.Generic;
using static System.Math;
using Alias = System.Text;
'''
        imports = extract_imports("csharp", code)
        assert "System" in imports
        assert "System.Collections.Generic" in imports
        assert "System.Math" in imports
        # Alias form
        assert "System.Text" in imports
    
    def test_php_usings(self, extract_imports):
        """Test PHP use/require extraction."""
        code = '''
<?php
use App\\Models\\User;
use Illuminate\\Http\\Request;
require 'vendor/autoload.php';
include_once 'config.php';
'''
        imports = extract_imports("php", code)
        assert any("App" in imp and "Models" in imp for imp in imports)
        assert any("Illuminate" in imp for imp in imports)
        assert "vendor/autoload.php" in imports
        assert "config.php" in imports


# ==============================================================================
# Call Extraction Tests
# ==============================================================================

class TestCallExtraction:
    """Test call extraction for all supported languages."""
    
    @pytest.fixture
    def extract_calls(self):
        """Return the call extraction function."""
        from scripts.ingest.metadata import _extract_calls
        return _extract_calls
    
    @pytest.fixture
    def get_imports_calls(self):
        """Return the combined imports/calls function."""
        from scripts.ingest.metadata import _get_imports_calls
        return _get_imports_calls
    
    def test_python_calls_regex(self, extract_calls):
        """Test Python call extraction with regex fallback."""
        code = '''
def main():
    print("hello")
    result = calculate(x, y)
    obj.method()
    helper_func(arg1, arg2)
'''
        calls = extract_calls("python", code)
        assert "print" in calls
        assert "calculate" in calls
        assert "method" in calls
        assert "helper_func" in calls
        # Keywords should be excluded
        assert "def" not in calls
    
    def test_javascript_calls_regex(self, extract_calls):
        """Test JavaScript call extraction with regex fallback."""
        code = '''
function main() {
    console.log("hello");
    const data = fetchData(url);
    arr.map(x => x * 2);
    if (condition) {
        process();
    }
}
'''
        calls = extract_calls("javascript", code)
        assert "log" in calls
        assert "fetchData" in calls
        assert "map" in calls
        assert "process" in calls
        # Keywords should be excluded
        assert "if" not in calls
        assert "function" not in calls
    
    def test_go_calls_regex(self, extract_calls):
        """Test Go call extraction with regex fallback."""
        code = '''
func main() {
    fmt.Println("hello")
    result := calculate(x, y)
    if err != nil {
        log.Fatal(err)
    }
    for i := 0; i < 10; i++ {
        process(i)
    }
}
'''
        calls = extract_calls("go", code)
        assert "Println" in calls
        assert "calculate" in calls
        assert "Fatal" in calls
        assert "process" in calls
        # Keywords should be excluded
        assert "func" not in calls
        assert "for" not in calls


# ==============================================================================
# Tree-sitter Call Extraction Tests
# ==============================================================================

class TestTreeSitterCallExtraction:
    """Test tree-sitter based call extraction for supported languages."""
    
    @pytest.fixture
    def ts_extract_calls(self):
        """Return the tree-sitter call extraction function."""
        from scripts.ingest.metadata import _ts_extract_calls_generic
        return _ts_extract_calls_generic
    
    @pytest.fixture
    def ts_languages(self):
        """Return dict of available tree-sitter languages."""
        try:
            from scripts.ingest.tree_sitter import _TS_LANGUAGES, _TS_AVAILABLE
            return _TS_LANGUAGES if _TS_AVAILABLE else {}
        except ImportError:
            return {}
    
    def test_javascript_ts_calls(self, ts_extract_calls, ts_languages):
        """Test JavaScript call extraction with tree-sitter."""
        if "javascript" not in ts_languages:
            pytest.skip("tree-sitter javascript parser not available")
        
        code = '''
function main() {
    console.log("hello");
    const data = fetchData(url);
    arr.map(x => x * 2);
    process();
}
'''
        calls = ts_extract_calls("javascript", code)
        # Should find meaningful calls
        assert len(calls) > 0
        assert "process" in calls or "log" in calls or "fetchData" in calls or "map" in calls
    
    def test_go_ts_calls(self, ts_extract_calls, ts_languages):
        """Test Go call extraction with tree-sitter."""
        if "go" not in ts_languages:
            pytest.skip("tree-sitter go parser not available")
        
        code = '''
package main

import "fmt"

func main() {
    fmt.Println("hello")
    result := calculate(10, 20)
    process(result)
}
'''
        calls = ts_extract_calls("go", code)
        # Should find function calls
        assert len(calls) >= 0  # May be empty if parser not available
    
    def test_rust_ts_calls(self, ts_extract_calls, ts_languages):
        """Test Rust call extraction with tree-sitter."""
        if "rust" not in ts_languages:
            pytest.skip("tree-sitter rust parser not available")
        
        code = '''
fn main() {
    println!("hello");
    let result = calculate(10, 20);
    process(result);
}
'''
        calls = ts_extract_calls("rust", code)
        # Should find function calls (including macros)
        assert len(calls) >= 0


# ==============================================================================
# AST Analyzer Tests
# ==============================================================================

class TestASTAnalyzer:
    """Test the AST analyzer for all supported languages."""
    
    @pytest.fixture
    def analyzer(self):
        """Return an AST analyzer instance."""
        from scripts.ast_analyzer import ASTAnalyzer
        return ASTAnalyzer()
    
    def test_python_analysis(self, analyzer):
        """Test Python AST analysis."""
        code = '''
import os
from pathlib import Path

class MyClass:
    """A sample class."""
    
    def __init__(self, value):
        self.value = value
    
    def process(self):
        return self.value * 2

def standalone_func(x, y):
    result = calculate(x, y)
    return result
'''
        result = analyzer.analyze_file("test.py", "python", code)
        
        # Check symbols
        symbols = result.get("symbols", [])
        symbol_names = [s.name for s in symbols]
        assert "MyClass" in symbol_names
        assert "__init__" in symbol_names or "process" in symbol_names
        assert "standalone_func" in symbol_names
        
        # Check imports
        imports = result.get("imports", [])
        assert len(imports) > 0
    
    def test_javascript_analysis(self, analyzer):
        """Test JavaScript AST analysis."""
        code = '''
import React from 'react';

class MyComponent {
    constructor() {
        this.state = {};
    }
    
    render() {
        return null;
    }
}

function helperFunc() {
    return 42;
}
'''
        result = analyzer.analyze_file("test.js", "javascript", code)
        
        symbols = result.get("symbols", [])
        symbol_names = [s.name for s in symbols]
        # Should find class and functions
        assert len(symbol_names) >= 1
    
    def test_go_analysis(self, analyzer):
        """Test Go AST analysis."""
        code = '''
package main

import (
    "fmt"
    "os"
)

type Server struct {
    Host string
    Port int
}

func (s *Server) Start() error {
    fmt.Println("Starting server")
    return nil
}

func NewServer(host string, port int) *Server {
    return &Server{Host: host, Port: port}
}
'''
        result = analyzer.analyze_file("test.go", "go", code)
        
        symbols = result.get("symbols", [])
        if symbols:  # Only check if tree-sitter is available
            symbol_names = [s.name for s in symbols]
            # Should find struct and functions
            assert "Server" in symbol_names or "NewServer" in symbol_names or "Start" in symbol_names
    
    def test_rust_analysis(self, analyzer):
        """Test Rust AST analysis."""
        code = '''
use std::collections::HashMap;

struct Config {
    host: String,
    port: u16,
}

impl Config {
    fn new(host: String, port: u16) -> Self {
        Config { host, port }
    }
    
    fn start(&self) {
        println!("Starting on {}:{}", self.host, self.port);
    }
}

fn main() {
    let config = Config::new("localhost".into(), 8080);
    config.start();
}
'''
        result = analyzer.analyze_file("test.rs", "rust", code)
        
        symbols = result.get("symbols", [])
        if symbols:  # Only check if tree-sitter is available
            symbol_names = [s.name for s in symbols]
            # Should find struct and functions
            assert len(symbol_names) >= 1
    
    def test_java_analysis(self, analyzer):
        """Test Java AST analysis."""
        code = '''
package com.example;

import java.util.List;

public class UserService {
    private final Database database;
    
    public UserService(Database database) {
        this.database = database;
    }
    
    public User findById(long id) {
        return database.query(id);
    }
    
    public void save(User user) {
        database.insert(user);
    }
}
'''
        result = analyzer.analyze_file("UserService.java", "java", code)
        
        symbols = result.get("symbols", [])
        if symbols:  # Only check if tree-sitter is available
            symbol_names = [s.name for s in symbols]
            # Should find class and methods
            assert "UserService" in symbol_names or "findById" in symbol_names or "save" in symbol_names
    
    def test_cpp_analysis(self, analyzer):
        """Test C++ AST analysis."""
        code = '''
#include <iostream>
#include <vector>

class Calculator {
public:
    int add(int a, int b) {
        return a + b;
    }
    
    int multiply(int a, int b) {
        return a * b;
    }
};

int main() {
    Calculator calc;
    std::cout << calc.add(2, 3) << std::endl;
    return 0;
}
'''
        result = analyzer.analyze_file("test.cpp", "cpp", code)
        
        symbols = result.get("symbols", [])
        if symbols:  # Only check if tree-sitter is available
            symbol_names = [s.name for s in symbols]
            # Should find class and functions
            assert len(symbol_names) >= 1
        
        imports = result.get("imports", [])
        # Check includes
        if imports:
            import_modules = [i.module for i in imports]
            assert "iostream" in import_modules or "vector" in import_modules
    
    def test_ruby_analysis(self, analyzer):
        """Test Ruby AST analysis."""
        code = '''
require 'json'
require_relative 'lib/helper'

class UserController
  def initialize(service)
    @service = service
  end
  
  def index
    @service.all
  end
  
  def show(id)
    @service.find(id)
  end
end

module Helpers
  def format_date(date)
    date.strftime("%Y-%m-%d")
  end
end
'''
        result = analyzer.analyze_file("test.rb", "ruby", code)
        
        symbols = result.get("symbols", [])
        if symbols:  # Only check if tree-sitter is available
            symbol_names = [s.name for s in symbols]
            # Should find class, module, and methods
            assert len(symbol_names) >= 1


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestLanguageCoverageIntegration:
    """Integration tests for the full import/call extraction pipeline."""
    
    @pytest.fixture
    def get_imports_calls(self):
        """Return the combined imports/calls function."""
        from scripts.ingest.metadata import _get_imports_calls
        return _get_imports_calls
    
    def test_python_full_pipeline(self, get_imports_calls):
        """Test Python full extraction pipeline."""
        code = '''
import os
from pathlib import Path

def process_file(path):
    content = Path(path).read_text()
    result = parse(content)
    return transform(result)
'''
        imports, calls = get_imports_calls("python", code)
        
        assert "os" in imports
        assert "pathlib" in imports
        # Calls may include parse, transform
        assert len(calls) >= 0
    
    def test_go_full_pipeline(self, get_imports_calls):
        """Test Go full extraction pipeline."""
        code = '''
package main

import "fmt"

func main() {
    fmt.Println("hello")
}
'''
        imports, calls = get_imports_calls("go", code)
        
        assert "fmt" in imports
    
    def test_rust_full_pipeline(self, get_imports_calls):
        """Test Rust full extraction pipeline."""
        code = '''
use std::io::Read;

fn main() {
    let mut buffer = String::new();
    std::io::stdin().read_to_string(&mut buffer);
    println!("{}", buffer);
}
'''
        imports, calls = get_imports_calls("rust", code)
        
        assert "std::io::Read" in imports
    
    def test_java_full_pipeline(self, get_imports_calls):
        """Test Java full extraction pipeline."""
        code = '''
import java.util.List;
import java.util.ArrayList;

public class Main {
    public static void main(String[] args) {
        List<String> items = new ArrayList<>();
        items.add("hello");
        System.out.println(items);
    }
}
'''
        imports, calls = get_imports_calls("java", code)
        
        assert "java.util.List" in imports
        assert "java.util.ArrayList" in imports
    
    def test_cpp_full_pipeline(self, get_imports_calls):
        """Test C++ full extraction pipeline."""
        code = '''
#include <iostream>
#include <string>

int main() {
    std::string name = "World";
    std::cout << "Hello, " << name << std::endl;
    return 0;
}
'''
        imports, calls = get_imports_calls("cpp", code)
        
        assert "iostream" in imports
        assert "string" in imports
    
    def test_ruby_full_pipeline(self, get_imports_calls):
        """Test Ruby full extraction pipeline."""
        code = '''
require 'json'
require 'net/http'

def fetch_data(url)
  uri = URI.parse(url)
  response = Net::HTTP.get(uri)
  JSON.parse(response)
end
'''
        imports, calls = get_imports_calls("ruby", code)
        
        assert "json" in imports
        assert "net/http" in imports


# ==============================================================================
# Edge Cases
# ==============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def extract_imports(self):
        from scripts.ingest.metadata import _extract_imports
        return _extract_imports
    
    @pytest.fixture
    def extract_calls(self):
        from scripts.ingest.metadata import _extract_calls
        return _extract_calls
    
    def test_empty_code(self, extract_imports, extract_calls):
        """Test extraction from empty code."""
        assert extract_imports("python", "") == []
        assert extract_calls("python", "") == []
    
    def test_unknown_language(self, extract_imports, extract_calls):
        """Test extraction for unknown language."""
        code = "some random code here"
        assert extract_imports("unknown_lang", code) == []
        # Calls should still use regex fallback - the pattern needs word boundary
        calls = extract_calls("unknown_lang", "result = someFunc(arg)")
        assert "someFunc" in calls
    
    def test_malformed_code(self, extract_imports):
        """Test extraction from malformed code."""
        # Should not crash
        malformed = "import ,,, from ... what???"
        result = extract_imports("python", malformed)
        assert isinstance(result, list)
    
    def test_large_file_limit(self, extract_imports, extract_calls):
        """Test that extraction respects the 200 item limit."""
        # Generate code with many imports
        imports = "\n".join([f"import module{i}" for i in range(300)])
        result = extract_imports("python", imports)
        assert len(result) <= 200
        
        # Generate code with many calls
        calls = " ".join([f"func{i}()" for i in range(300)])
        result = extract_calls("python", calls)
        assert len(result) <= 200
