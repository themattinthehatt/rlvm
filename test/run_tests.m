% Script to run unit tests on XAutoencoder. Requires MATLAB >= R2013a.

% import package for test suite to use w/o package qualifier
import matlab.unittest.TestSuite

% add tests folder to MATLAB path
% path = addpath(genpath(pwd));

% run all test suites located in the 'tests' directory
% results = run(TestSuite.fromFolder('tests'));

% run a test suite from 'tests' directory
results = run(TestSuite.fromFile('tests/RLVMTests.m'));
% results = run(TestSuite.fromFile('tests/AutoSubunitTests.m'));
% results = run(TestSuite.fromFile('tests/StimSubunitTests.m'));

% output results
disp(results)