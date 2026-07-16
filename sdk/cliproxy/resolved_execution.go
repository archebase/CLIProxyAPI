package cliproxy

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"

	nativeexecutor "github.com/router-for-me/CLIProxyAPI/v7/internal/runtime/executor"
	coreauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
)

type NativeExecutorKind string

const (
	NativeExecutorOpenAICompatible NativeExecutorKind = "openai-compatible"
	NativeExecutorAnthropic        NativeExecutorKind = "anthropic"
)

type ResolvedExecutionCandidate struct {
	Auth         *coreauth.Auth
	RuntimeModel string
	APIBase      string
}

type ResolvedExecutionRequest struct {
	Executor   NativeExecutorKind
	Provider   string
	Candidates []ResolvedExecutionCandidate
	Request    cliproxyexecutor.Request
	Options    cliproxyexecutor.Options
}

func (r *EmbeddedRuntime) ExecuteResolved(ctx context.Context, execution ResolvedExecutionRequest) (cliproxyexecutor.Response, error) {
	ctx, finish, err := r.beginResolvedExecution(ctx)
	if err != nil {
		return cliproxyexecutor.Response{}, err
	}
	defer finish()
	executor, provider, candidates, err := r.prepareResolvedCandidates(execution)
	if err != nil {
		return cliproxyexecutor.Response{}, err
	}
	var lastErr error
	for _, candidate := range candidates {
		req := execution.Request
		req.Model = candidate.RuntimeModel
		resp, errExec := executor.Execute(ctx, candidate.Auth, req, execution.Options)
		if errExec == nil {
			return resp, nil
		}
		if err := ctx.Err(); err != nil {
			return cliproxyexecutor.Response{}, err
		}
		if resolvedRequestInvalid(errExec) {
			return cliproxyexecutor.Response{}, errExec
		}
		lastErr = errExec
	}
	if lastErr != nil {
		return cliproxyexecutor.Response{}, lastErr
	}
	return cliproxyexecutor.Response{}, &coreauth.Error{Code: "auth_not_found", Message: fmt.Sprintf("no %s auth available", provider)}
}

func (r *EmbeddedRuntime) ExecuteResolvedStream(ctx context.Context, execution ResolvedExecutionRequest) (*cliproxyexecutor.StreamResult, error) {
	ctx, finish, err := r.beginResolvedExecution(ctx)
	if err != nil {
		return nil, err
	}
	executor, provider, candidates, err := r.prepareResolvedCandidates(execution)
	if err != nil {
		finish()
		return nil, err
	}
	var lastErr error
	for _, candidate := range candidates {
		req := execution.Request
		req.Model = candidate.RuntimeModel
		result, errExec := executor.ExecuteStream(ctx, candidate.Auth, req, execution.Options)
		if errExec == nil {
			first, ok := <-result.Chunks
			if ok && first.Err != nil {
				if resolvedRequestInvalid(first.Err) {
					finish()
					return nil, first.Err
				}
				lastErr = first.Err
				continue
			}
			forwarded := make(chan cliproxyexecutor.StreamChunk)
			go func() {
				defer close(forwarded)
				defer finish()
				if ok {
					select {
					case forwarded <- first:
					case <-ctx.Done():
						return
					}
				}
				for chunk := range result.Chunks {
					select {
					case forwarded <- chunk:
					case <-ctx.Done():
						return
					}
				}
			}()
			return &cliproxyexecutor.StreamResult{Headers: result.Headers, Chunks: forwarded}, nil
		}
		if err := ctx.Err(); err != nil {
			finish()
			return nil, err
		}
		if resolvedRequestInvalid(errExec) {
			finish()
			return nil, errExec
		}
		lastErr = errExec
	}
	finish()
	if lastErr != nil {
		return nil, lastErr
	}
	return nil, &coreauth.Error{Code: "auth_not_found", Message: fmt.Sprintf("no %s auth available", provider)}
}

type preparedResolvedCandidate struct {
	Auth         *coreauth.Auth
	RuntimeModel string
}

func (r *EmbeddedRuntime) prepareResolvedCandidates(execution ResolvedExecutionRequest) (coreauth.ProviderExecutor, string, []preparedResolvedCandidate, error) {
	if len(execution.Candidates) == 0 {
		return nil, "", nil, &coreauth.Error{Code: "auth_not_found", Message: "no resolved auth candidates"}
	}
	provider := strings.ToLower(strings.TrimSpace(execution.Provider))
	if provider == "" {
		provider = "resolved"
	}
	var executor coreauth.ProviderExecutor
	switch execution.Executor {
	case NativeExecutorOpenAICompatible:
		executor = nativeexecutor.NewOpenAICompatExecutor(provider, r.service.cfg)
	case NativeExecutorAnthropic:
		provider = "claude"
		executor = nativeexecutor.NewClaudeExecutor(r.service.cfg)
	default:
		return nil, "", nil, fmt.Errorf("cliproxy: unsupported resolved executor %q", execution.Executor)
	}
	candidates := make([]preparedResolvedCandidate, 0, len(execution.Candidates))
	for index, candidate := range execution.Candidates {
		if candidate.Auth == nil {
			return nil, "", nil, fmt.Errorf("cliproxy: resolved auth candidate %d is required", index)
		}
		model := strings.TrimSpace(candidate.RuntimeModel)
		if model == "" {
			return nil, "", nil, fmt.Errorf("cliproxy: resolved runtime model is required")
		}
		apiBase := resolvedExecutorBase(execution.Executor, candidate.APIBase)
		if apiBase == "" {
			return nil, "", nil, fmt.Errorf("cliproxy: resolved API base is required")
		}
		apiKey := strings.TrimSpace(candidate.Auth.Attributes["api_key"])
		if apiKey == "" {
			return nil, "", nil, fmt.Errorf("cliproxy: resolved API key is required")
		}
		auth := &coreauth.Auth{
			ID:       fmt.Sprintf("resolved-%d", index),
			Provider: provider,
			Label:    candidate.Auth.Label,
			Status:   coreauth.StatusActive,
			ProxyURL: candidate.Auth.ProxyURL,
			Attributes: map[string]string{
				"api_key":  apiKey,
				"base_url": apiBase,
			},
		}
		candidates = append(candidates, preparedResolvedCandidate{Auth: auth, RuntimeModel: model})
	}
	return executor, provider, candidates, nil
}

func (r *EmbeddedRuntime) beginResolvedExecution(ctx context.Context) (context.Context, func(), error) {
	if r == nil || r.service == nil || r.service.cfg == nil {
		return nil, nil, fmt.Errorf("cliproxy: embedded runtime is unavailable")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	r.mu.Lock()
	started, closed := r.started, r.closed
	r.mu.Unlock()
	if !started || closed {
		return nil, nil, fmt.Errorf("cliproxy: embedded runtime is not started")
	}
	r.resolvedMu.Lock()
	if r.resolvedClosing {
		r.resolvedMu.Unlock()
		return nil, nil, fmt.Errorf("cliproxy: embedded runtime is closing")
	}
	r.resolvedInflight++
	r.resolvedMu.Unlock()
	return ctx, r.finishResolvedExecution, nil
}

func (r *EmbeddedRuntime) finishResolvedExecution() {
	r.resolvedMu.Lock()
	r.resolvedInflight--
	if r.resolvedClosing && r.resolvedInflight == 0 && r.resolvedDrained != nil {
		close(r.resolvedDrained)
		r.resolvedDrained = nil
	}
	r.resolvedMu.Unlock()
}

func (r *EmbeddedRuntime) closeResolvedExecutions(ctx context.Context) error {
	r.resolvedMu.Lock()
	r.resolvedClosing = true
	if r.resolvedInflight == 0 {
		r.resolvedMu.Unlock()
		return nil
	}
	if r.resolvedDrained == nil {
		r.resolvedDrained = make(chan struct{})
	}
	drained := r.resolvedDrained
	r.resolvedMu.Unlock()
	select {
	case <-drained:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func resolvedExecutorBase(kind NativeExecutorKind, apiBase string) string {
	apiBase = strings.TrimRight(strings.TrimSpace(apiBase), "/")
	if kind == NativeExecutorAnthropic {
		apiBase = strings.TrimSuffix(apiBase, "/v1")
	}
	return apiBase
}

func resolvedRequestInvalid(err error) bool {
	var scoped interface{ IsRequestScoped() bool }
	if errors.As(err, &scoped) && scoped.IsRequestScoped() {
		return true
	}
	var status interface{ StatusCode() int }
	if errors.As(err, &status) {
		return status.StatusCode() == http.StatusBadRequest || status.StatusCode() == http.StatusUnprocessableEntity
	}
	return false
}
