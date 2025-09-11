import * as vscode from 'vscode';
import { LanguageClient, LanguageClientOptions, ServerOptions } from 'vscode-languageclient/node';
import * as path from 'path';
import * as fs from 'fs';

let client: LanguageClient;

export function activate(context: vscode.ExtensionContext) {
    console.log('SynthLang extension is now active');

    // Register language server
    if (vscode.workspace.getConfiguration('synthlang').get('enableLSP')) {
        activateLanguageServer(context);
    }

    // Register commands
    registerCommands(context);

    // Register providers
    registerProviders(context);

    // Status bar item
    const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    statusBar.text = "$(play) SynthLang";
    statusBar.tooltip = "SynthLang Pipeline Tools";
    statusBar.command = 'synthlang.showCommands';
    statusBar.show();
    context.subscriptions.push(statusBar);
}

function activateLanguageServer(context: vscode.ExtensionContext) {
    const config = vscode.workspace.getConfiguration('synthlang');
    const compilerPath = config.get<string>('compiler.path', 'synth');

    // Check if LSP server exists
    const serverPath = path.join(compilerPath, 'lsp');
    
    const serverOptions: ServerOptions = {
        run: { command: serverPath, args: ['--lsp'] },
        debug: { command: serverPath, args: ['--lsp', '--debug'] }
    };

    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'synth' }],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.synth')
        }
    };

    client = new LanguageClient(
        'synthlang',
        'SynthLang Language Server',
        serverOptions,
        clientOptions
    );

    client.start();
    context.subscriptions.push(client);
}

function registerCommands(context: vscode.ExtensionContext) {
    // Run Pipeline command
    const runPipeline = vscode.commands.registerCommand('synthlang.runPipeline', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document.languageId !== 'synth') {
            vscode.window.showErrorMessage('No SynthLang file is open');
            return;
        }

        const document = editor.document;
        await document.save();

        try {
            const terminal = vscode.window.createTerminal('SynthLang');
            terminal.sendText(`synth run "${document.fileName}"`);
            terminal.show();

            vscode.window.showInformationMessage('Pipeline execution started');
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to run pipeline: ${error}`);
        }
    });

    // Validate Pipeline command
    const validatePipeline = vscode.commands.registerCommand('synthlang.validatePipeline', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document.languageId !== 'synth') {
            vscode.window.showErrorMessage('No SynthLang file is open');
            return;
        }

        const document = editor.document;
        await document.save();

        try {
            const config = vscode.workspace.getConfiguration('synthlang');
            const compilerPath = config.get<string>('compiler.path', 'synth');
            
            const result = await runCommand(compilerPath, ['validate', document.fileName]);
            
            if (result.success) {
                vscode.window.showInformationMessage('Pipeline validation passed ✓');
            } else {
                vscode.window.showErrorMessage(`Validation failed: ${result.error}`);
            }
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to validate pipeline: ${error}`);
        }
    });

    // Run Evaluation command
    const runEvaluation = vscode.commands.registerCommand('synthlang.runEvaluation', async () => {
        const datasets = await findEvalDatasets();
        
        if (datasets.length === 0) {
            vscode.window.showWarningMessage('No evaluation datasets found');
            return;
        }

        const selectedDataset = await vscode.window.showQuickPick(datasets, {
            placeHolder: 'Select evaluation dataset'
        });

        if (!selectedDataset) return;

        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document.languageId !== 'synth') {
            vscode.window.showErrorMessage('No SynthLang pipeline is open');
            return;
        }

        await document.save();

        try {
            const terminal = vscode.window.createTerminal('SynthLang Eval');
            terminal.sendText(`synth eval "${editor.document.fileName}" --dataset "${selectedDataset}"`);
            terminal.show();

            vscode.window.showInformationMessage('Evaluation started');
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to run evaluation: ${error}`);
        }
    });

    // Show Metrics command
    const showMetrics = vscode.commands.registerCommand('synthlang.showMetrics', async () => {
        const panel = vscode.window.createWebviewPanel(
            'synthMetrics',
            'Pipeline Metrics',
            vscode.ViewColumn.Two,
            {
                enableScripts: true,
                localResourceRoots: [context.extensionUri]
            }
        );

        panel.webview.html = getMetricsWebviewContent();

        // Load metrics data
        try {
            const metricsData = await loadMetrics();
            panel.webview.postMessage({ type: 'metricsData', data: metricsData });
        } catch (error) {
            console.error('Failed to load metrics:', error);
        }
    });

    // Deploy Pipeline command
    const deployPipeline = vscode.commands.registerCommand('synthlang.deployPipeline', async () => {
        const deployTargets = ['local', 'cloud', 'edge'];
        const config = vscode.workspace.getConfiguration('synthlang');
        const defaultTarget = config.get<string>('deployment.defaultTarget', 'local');

        const selectedTarget = await vscode.window.showQuickPick(deployTargets, {
            placeHolder: 'Select deployment target',
            canPickMany: false
        });

        if (!selectedTarget) return;

        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document.languageId !== 'synth') {
            vscode.window.showErrorMessage('No SynthLang pipeline is open');
            return;
        }

        await editor.document.save();

        try {
            const terminal = vscode.window.createTerminal('SynthLang Deploy');
            terminal.sendText(`synth deploy "${editor.document.fileName}" --target ${selectedTarget}`);
            terminal.show();

            vscode.window.showInformationMessage(`Deployment to ${selectedTarget} started`);
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to deploy pipeline: ${error}`);
        }
    });

    // Show Commands command
    const showCommands = vscode.commands.registerCommand('synthlang.showCommands', async () => {
        const commands = [
            { label: '$(play) Run Pipeline', command: 'synthlang.runPipeline' },
            { label: '$(check) Validate Pipeline', command: 'synthlang.validatePipeline' },
            { label: '$(graph) Run Evaluation', command: 'synthlang.runEvaluation' },
            { label: '$(dashboard) Show Metrics', command: 'synthlang.showMetrics' },
            { label: '$(cloud-upload) Deploy Pipeline', command: 'synthlang.deployPipeline' },
        ];

        const selected = await vscode.window.showQuickPick(commands, {
            placeHolder: 'SynthLang Commands'
        });

        if (selected) {
            vscode.commands.executeCommand(selected.command);
        }
    });

    context.subscriptions.push(
        runPipeline,
        validatePipeline,
        runEvaluation,
        showMetrics,
        deployPipeline,
        showCommands
    );
}

function registerProviders(context: vscode.ExtensionContext) {
    // Code completion provider
    const completionProvider = vscode.languages.registerCompletionItemProvider(
        'synth',
        {
            provideCompletionItems(document: vscode.TextDocument, position: vscode.Position) {
                const completions: vscode.CompletionItem[] = [];

                // Pipeline structure completions
                const pipelineCompletion = new vscode.CompletionItem('pipeline', vscode.CompletionItemKind.Keyword);
                pipelineCompletion.insertText = new vscode.SnippetString('pipeline ${1:PipelineName} {\n\t$0\n}');
                pipelineCompletion.documentation = 'Define a new pipeline';
                completions.push(pipelineCompletion);

                // Node type completions
                const nodeTypes = ['model', 'prompt', 'router', 'guardrail', 'cache', 'evaluator'];
                nodeTypes.forEach(nodeType => {
                    const completion = new vscode.CompletionItem(nodeType, vscode.CompletionItemKind.Class);
                    completion.insertText = new vscode.SnippetString(`${nodeType} \${1:${nodeType}_name} {\n\t$0\n}`);
                    completion.documentation = `Define a ${nodeType} node`;
                    completions.push(completion);
                });

                // Model providers
                const providers = ['openai', 'anthropic', 'huggingface', 'local'];
                providers.forEach(provider => {
                    const completion = new vscode.CompletionItem(provider, vscode.CompletionItemKind.Value);
                    completion.insertText = `"${provider}"`;
                    completion.documentation = `${provider} model provider`;
                    completions.push(completion);
                });

                return completions;
            }
        },
        ' ', '\t', '\n'
    );

    // Hover provider
    const hoverProvider = vscode.languages.registerHoverProvider('synth', {
        provideHover(document, position, token) {
            const range = document.getWordRangeAtPosition(position);
            const word = document.getText(range);

            const hoverInfo = getHoverInfo(word);
            if (hoverInfo) {
                return new vscode.Hover(hoverInfo);
            }
        }
    });

    context.subscriptions.push(completionProvider, hoverProvider);
}

async function runCommand(command: string, args: string[]): Promise<{ success: boolean; error?: string }> {
    return new Promise((resolve) => {
        const { spawn } = require('child_process');
        const process = spawn(command, args);
        
        let error = '';
        process.stderr.on('data', (data: Buffer) => {
            error += data.toString();
        });
        
        process.on('close', (code: number) => {
            resolve({
                success: code === 0,
                error: code !== 0 ? error : undefined
            });
        });
    });
}

async function findEvalDatasets(): Promise<string[]> {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders) return [];

    const datasets: string[] = [];
    for (const folder of workspaceFolders) {
        try {
            const evalDir = path.join(folder.uri.fsPath, 'evals');
            if (fs.existsSync(evalDir)) {
                const files = fs.readdirSync(evalDir);
                const jsonFiles = files.filter(f => f.endsWith('.json') || f.endsWith('.jsonl'));
                datasets.push(...jsonFiles.map(f => path.join(evalDir, f)));
            }
        } catch (error) {
            console.error('Error finding eval datasets:', error);
        }
    }
    return datasets;
}

async function loadMetrics(): Promise<any> {
    // Load metrics from metrics endpoint or local files
    const config = vscode.workspace.getConfiguration('synthlang');
    const endpoint = config.get<string>('metrics.endpoint');
    
    if (endpoint) {
        // Load from API endpoint
        try {
            const response = await fetch(endpoint);
            return await response.json();
        } catch (error) {
            console.error('Failed to load metrics from endpoint:', error);
        }
    }
    
    // Load from local metrics file
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (workspaceFolders) {
        const metricsPath = path.join(workspaceFolders[0].uri.fsPath, 'metrics.json');
        if (fs.existsSync(metricsPath)) {
            const content = fs.readFileSync(metricsPath, 'utf8');
            return JSON.parse(content);
        }
    }
    
    return {
        pipelines: [],
        evaluations: [],
        deployments: []
    };
}

function getMetricsWebviewContent(): string {
    return `
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Pipeline Metrics</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                margin: 0;
                padding: 20px;
                background-color: var(--vscode-editor-background);
                color: var(--vscode-editor-foreground);
            }
            .metric-card {
                background: var(--vscode-input-background);
                border: 1px solid var(--vscode-input-border);
                border-radius: 6px;
                padding: 16px;
                margin-bottom: 16px;
            }
            .metric-title {
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 12px;
                color: var(--vscode-textLink-foreground);
            }
            .metric-value {
                font-size: 24px;
                font-weight: 700;
                color: var(--vscode-charts-green);
            }
            .metric-description {
                font-size: 14px;
                color: var(--vscode-descriptionForeground);
                margin-top: 4px;
            }
            .chart-container {
                height: 200px;
                background: var(--vscode-editor-background);
                border: 1px solid var(--vscode-panel-border);
                border-radius: 4px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: var(--vscode-descriptionForeground);
            }
        </style>
    </head>
    <body>
        <h1>SynthLang Pipeline Metrics</h1>
        
        <div class="metric-card">
            <div class="metric-title">Active Pipelines</div>
            <div class="metric-value" id="activePipelines">-</div>
            <div class="metric-description">Currently running pipelines</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">Total Requests</div>
            <div class="metric-value" id="totalRequests">-</div>
            <div class="metric-description">Requests processed today</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">Average Latency</div>
            <div class="metric-value" id="avgLatency">-</div>
            <div class="metric-description">Response time (ms)</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">Cost Today</div>
            <div class="metric-value" id="costToday">-</div>
            <div class="metric-description">API costs in USD</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">Success Rate</div>
            <div class="metric-value" id="successRate">-</div>
            <div class="metric-description">Successful requests (%)</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">Performance Trend</div>
            <div class="chart-container">
                Chart visualization would go here
            </div>
        </div>
        
        <script>
            const vscode = acquireVsCodeApi();
            
            window.addEventListener('message', event => {
                const message = event.data;
                if (message.type === 'metricsData') {
                    updateMetrics(message.data);
                }
            });
            
            function updateMetrics(data) {
                document.getElementById('activePipelines').textContent = data.activePipelines || 0;
                document.getElementById('totalRequests').textContent = data.totalRequests || 0;
                document.getElementById('avgLatency').textContent = (data.avgLatency || 0) + ' ms';
                document.getElementById('costToday').textContent = '$' + (data.costToday || 0).toFixed(2);
                document.getElementById('successRate').textContent = ((data.successRate || 0) * 100).toFixed(1) + '%';
            }
        </script>
    </body>
    </html>
    `;
}

function getHoverInfo(word: string): string | undefined {
    const hoverMap: { [key: string]: string } = {
        'pipeline': 'A pipeline defines a directed graph of AI operations and transformations',
        'model': 'A model node represents an AI model endpoint (LLM, embedding, etc.)',
        'prompt': 'A prompt template with variable substitution support',
        'router': 'Routes requests based on conditions, A/B testing, or load balancing',
        'guardrail': 'Safety checks for content filtering, PII detection, etc.',
        'cache': 'Response caching to improve performance and reduce costs',
        'evaluator': 'Evaluation metrics for model quality assessment',
        'eval': 'Evaluation harness for testing pipeline performance',
        'temperature': 'Controls randomness in model output (0.0 = deterministic, 1.0 = random)',
        'max_tokens': 'Maximum number of tokens to generate',
        'provider': 'AI model provider (openai, anthropic, huggingface, etc.)'
    };
    
    return hoverMap[word.toLowerCase()];
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}