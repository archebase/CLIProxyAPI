package cliproxy

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/registry"
	coreauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/config"
	cliproxytranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
)

func TestEmbeddedRuntimeExecuteResolvedUsesNativeRetryAndExactModel(t *testing.T) {
	var firstCalls, secondCalls atomic.Int32
	first := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		firstCalls.Add(1)
		http.Error(w, `{"error":"unauthorized"}`, http.StatusUnauthorized)
	}))
	defer first.Close()
	second := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		secondCalls.Add(1)
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), `"model":"deepseek-chat"`) {
			t.Fatalf("body=%s, want exact runtime model", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-ok","choices":[]}`))
	}))
	defer second.Close()

	runtime := resolvedTestRuntime(t)
	resp, err := runtime.ExecuteResolved(context.Background(), ResolvedExecutionRequest{
		Executor: NativeExecutorOpenAICompatible,
		Provider: "deepseek",
		Candidates: []ResolvedExecutionCandidate{
			{Auth: &coreauth.Auth{ID: "first", Attributes: map[string]string{"api_key": "one"}}, RuntimeModel: "deepseek-chat", APIBase: first.URL},
			{Auth: &coreauth.Auth{ID: "second", Attributes: map[string]string{"api_key": "two"}}, RuntimeModel: "deepseek-chat", APIBase: second.URL},
		},
		Request: cliproxyexecutor.Request{Payload: []byte(`{"messages":[]}`), Format: cliproxytranslator.FormatOpenAI},
		Options: cliproxyexecutor.Options{SourceFormat: cliproxytranslator.FormatOpenAI, ResponseFormat: cliproxytranslator.FormatOpenAI},
	})
	if err != nil {
		t.Fatalf("ExecuteResolved: %v", err)
	}
	if !strings.Contains(string(resp.Payload), "chatcmpl-ok") || firstCalls.Load() != 1 || secondCalls.Load() != 1 {
		t.Fatalf("payload=%s calls=%d/%d", resp.Payload, firstCalls.Load(), secondCalls.Load())
	}
}

func TestEmbeddedRuntimeExecuteResolvedStopsOnClientRequestError(t *testing.T) {
	var secondCalls atomic.Int32
	first := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, `{"error":"bad request"}`, http.StatusBadRequest)
	}))
	defer first.Close()
	second := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		secondCalls.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer second.Close()

	runtime := resolvedTestRuntime(t)
	_, err := runtime.ExecuteResolved(t.Context(), ResolvedExecutionRequest{
		Executor: NativeExecutorOpenAICompatible,
		Provider: "deepseek",
		Candidates: []ResolvedExecutionCandidate{
			{Auth: &coreauth.Auth{ID: "first", Attributes: map[string]string{"api_key": "one"}}, RuntimeModel: "deepseek-chat", APIBase: first.URL},
			{Auth: &coreauth.Auth{ID: "second", Attributes: map[string]string{"api_key": "two"}}, RuntimeModel: "deepseek-chat", APIBase: second.URL},
		},
		Request: cliproxyexecutor.Request{Payload: []byte(`{"messages":[]}`), Format: cliproxytranslator.FormatOpenAI},
		Options: cliproxyexecutor.Options{SourceFormat: cliproxytranslator.FormatOpenAI, ResponseFormat: cliproxytranslator.FormatOpenAI},
	})
	if err == nil || secondCalls.Load() != 0 {
		t.Fatalf("error=%v second_calls=%d, want terminal client error", err, secondCalls.Load())
	}
}

func TestEmbeddedRuntimeExecuteResolvedAnthropicUsesExactVersionedPrefix(t *testing.T) {
	var path string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg-ok","type":"message","content":[]}`))
	}))
	defer upstream.Close()

	runtime := resolvedTestRuntime(t)
	resp, err := runtime.ExecuteResolved(t.Context(), ResolvedExecutionRequest{
		Executor: NativeExecutorAnthropic,
		Provider: "zhipu",
		Candidates: []ResolvedExecutionCandidate{{
			Auth:         &coreauth.Auth{ID: "zhipu", Attributes: map[string]string{"api_key": "key"}},
			RuntimeModel: "glm-5.2",
			APIBase:      upstream.URL + "/api/anthropic/v1",
		}},
		Request: cliproxyexecutor.Request{Payload: []byte(`{"messages":[],"max_tokens":16}`), Format: cliproxytranslator.FormatClaude},
		Options: cliproxyexecutor.Options{SourceFormat: cliproxytranslator.FormatClaude, ResponseFormat: cliproxytranslator.FormatClaude},
	})
	if err != nil {
		t.Fatalf("ExecuteResolved: %v", err)
	}
	if path != "/api/anthropic/v1/messages" || !strings.Contains(string(resp.Payload), "msg-ok") {
		t.Fatalf("path=%q payload=%s", path, resp.Payload)
	}
}

func TestEmbeddedRuntimeExecuteResolvedStreamRetriesBeforePayloadWithoutServer(t *testing.T) {
	var secondCalls atomic.Int32
	first := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, `{"error":"unauthorized"}`, http.StatusUnauthorized)
	}))
	defer first.Close()
	second := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		secondCalls.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-stream\",\"choices\":[]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer second.Close()

	runtime := resolvedTestRuntime(t)
	result, err := runtime.ExecuteResolvedStream(t.Context(), ResolvedExecutionRequest{
		Executor: NativeExecutorOpenAICompatible,
		Provider: "deepseek",
		Candidates: []ResolvedExecutionCandidate{
			{Auth: &coreauth.Auth{ID: "first", Attributes: map[string]string{"api_key": "one"}}, RuntimeModel: "deepseek-chat", APIBase: first.URL},
			{Auth: &coreauth.Auth{ID: "second", Attributes: map[string]string{"api_key": "two"}}, RuntimeModel: "deepseek-chat", APIBase: second.URL},
		},
		Request: cliproxyexecutor.Request{Payload: []byte(`{"messages":[],"stream":true}`), Format: cliproxytranslator.FormatOpenAI},
		Options: cliproxyexecutor.Options{Stream: true, SourceFormat: cliproxytranslator.FormatOpenAI, ResponseFormat: cliproxytranslator.FormatOpenAI},
	})
	if err != nil {
		t.Fatalf("ExecuteResolvedStream: %v", err)
	}
	var payload strings.Builder
	for chunk := range result.Chunks {
		if chunk.Err != nil {
			t.Fatalf("stream chunk: %v", chunk.Err)
		}
		payload.Write(chunk.Payload)
	}
	if secondCalls.Load() != 1 || !strings.Contains(payload.String(), "chatcmpl-stream") {
		t.Fatalf("calls=%d payload=%q", secondCalls.Load(), payload.String())
	}
	if runtime.ServerStarted() || runtime.WatcherStarted() {
		t.Fatal("resolved execution started CLIProxyAPI server or watcher")
	}
}

func TestEmbeddedRuntimeExecuteResolvedRejectsEmptyAnthropicBase(t *testing.T) {
	runtime := resolvedTestRuntime(t)
	_, err := runtime.ExecuteResolved(t.Context(), ResolvedExecutionRequest{
		Executor:   NativeExecutorAnthropic,
		Provider:   "zhipu",
		Candidates: []ResolvedExecutionCandidate{{Auth: &coreauth.Auth{Attributes: map[string]string{"api_key": "secret"}}, RuntimeModel: "glm-5.2"}},
		Request:    cliproxyexecutor.Request{Payload: []byte(`{"messages":[]}`), Format: cliproxytranslator.FormatClaude},
		Options:    cliproxyexecutor.Options{SourceFormat: cliproxytranslator.FormatClaude, ResponseFormat: cliproxytranslator.FormatClaude},
	})
	if err == nil || !strings.Contains(err.Error(), "API base") {
		t.Fatalf("error=%v, want API base validation", err)
	}
}

func TestEmbeddedRuntimeExecuteResolvedDoesNotMutateGlobalModelRegistry(t *testing.T) {
	before := len(registry.GetGlobalRegistry().GetAvailableModels("openai"))
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"ok","choices":[]}`))
	}))
	defer upstream.Close()
	runtime := resolvedTestRuntime(t)
	_, err := runtime.ExecuteResolved(t.Context(), ResolvedExecutionRequest{
		Executor:   NativeExecutorOpenAICompatible,
		Provider:   "deepseek",
		Candidates: []ResolvedExecutionCandidate{{Auth: &coreauth.Auth{Attributes: map[string]string{"api_key": "secret"}}, RuntimeModel: "deepseek-chat", APIBase: upstream.URL}},
		Request:    cliproxyexecutor.Request{Payload: []byte(`{"messages":[]}`), Format: cliproxytranslator.FormatOpenAI},
		Options:    cliproxyexecutor.Options{SourceFormat: cliproxytranslator.FormatOpenAI, ResponseFormat: cliproxytranslator.FormatOpenAI},
	})
	if err != nil {
		t.Fatalf("ExecuteResolved: %v", err)
	}
	after := len(registry.GetGlobalRegistry().GetAvailableModels("openai"))
	if after != before {
		t.Fatalf("global model count=%d, want %d", after, before)
	}
}

func TestEmbeddedRuntimeCloseWaitsForResolvedStreamAndRejectsNewCalls(t *testing.T) {
	release := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"started\",\"choices\":[]}\n\n"))
		w.(http.Flusher).Flush()
		<-release
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer upstream.Close()
	runtime := resolvedTestRuntime(t)
	result, err := runtime.ExecuteResolvedStream(t.Context(), ResolvedExecutionRequest{
		Executor:   NativeExecutorOpenAICompatible,
		Provider:   "deepseek",
		Candidates: []ResolvedExecutionCandidate{{Auth: &coreauth.Auth{Attributes: map[string]string{"api_key": "secret"}}, RuntimeModel: "deepseek-chat", APIBase: upstream.URL}},
		Request:    cliproxyexecutor.Request{Payload: []byte(`{"messages":[],"stream":true}`), Format: cliproxytranslator.FormatOpenAI},
		Options:    cliproxyexecutor.Options{Stream: true, SourceFormat: cliproxytranslator.FormatOpenAI, ResponseFormat: cliproxytranslator.FormatOpenAI},
	})
	if err != nil {
		t.Fatalf("ExecuteResolvedStream: %v", err)
	}
	closeCtx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if err := runtime.Close(closeCtx); err == nil {
		t.Fatal("Close returned before resolved stream drained")
	}
	_, err = runtime.ExecuteResolved(t.Context(), ResolvedExecutionRequest{})
	if err == nil || !strings.Contains(err.Error(), "closing") {
		t.Fatalf("post-close-start error=%v", err)
	}
	close(release)
	for range result.Chunks {
	}
	if err := runtime.Close(context.Background()); err != nil {
		t.Fatalf("retry Close: %v", err)
	}
}

func TestEmbeddedRuntimeExecuteResolvedAppliesInterceptorAndHostRoundTripper(t *testing.T) {
	var gotBody string
	var gotTransportAuth *coreauth.Auth
	var gotIntercept cliproxyexecutor.RequestAfterAuthInterceptRequest
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		gotBody = string(body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"ok","choices":[]}`))
	}))
	defer upstream.Close()
	upstreamURL, _ := url.Parse(upstream.URL)
	manager := coreauth.NewManager(nil, nil, nil)
	manager.SetRoundTripperProvider(roundTripperProviderFunc(func(auth *coreauth.Auth) http.RoundTripper {
		gotTransportAuth = auth.Clone()
		return roundTripperFunc(func(req *http.Request) (*http.Response, error) {
			clone := req.Clone(req.Context())
			clone.URL.Scheme = upstreamURL.Scheme
			clone.URL.Host = upstreamURL.Host
			return http.DefaultTransport.RoundTrip(clone)
		})
	}))
	runtime, err := NewEmbeddedRuntime(&config.Config{}, manager)
	if err != nil {
		t.Fatalf("NewEmbeddedRuntime: %v", err)
	}
	if err := runtime.Start(t.Context()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	callerHeaders := http.Header{"X-Caller": []string{"preserved"}}
	callerMetadata := map[string]any{
		cliproxyexecutor.RequestedModelMetadataKey: "canonical-model",
		"project_id": "project-1",
	}
	candidateAuth := &coreauth.Auth{
		ID:       "credential-17",
		Provider: "deepseek",
		Metadata: map[string]any{"tenant": "tenant-1"},
		Attributes: map[string]string{
			"api_key":     "secret",
			"egress_zone": "cn-east",
		},
	}
	_, err = runtime.ExecuteResolved(t.Context(), ResolvedExecutionRequest{
		Executor:   NativeExecutorOpenAICompatible,
		Provider:   "deepseek",
		Candidates: []ResolvedExecutionCandidate{{Auth: candidateAuth, RuntimeModel: "deepseek-chat", APIBase: "https://blocked.invalid"}},
		Request:    cliproxyexecutor.Request{Model: "request-model", Payload: []byte(`{"messages":[]}`), Format: cliproxytranslator.FormatOpenAI},
		Options: cliproxyexecutor.Options{
			SourceFormat:   cliproxytranslator.FormatOpenAI,
			ResponseFormat: cliproxytranslator.FormatOpenAI,
			Headers:        callerHeaders,
			Metadata:       callerMetadata,
			RequestAfterAuthInterceptor: func(_ context.Context, input cliproxyexecutor.RequestAfterAuthInterceptRequest) cliproxyexecutor.RequestAfterAuthInterceptResponse {
				gotIntercept = input
				input.Metadata["project_id"] = "mutated"
				return cliproxyexecutor.RequestAfterAuthInterceptResponse{Headers: http.Header{"X-Resolved-Policy": []string{"applied"}}, Body: []byte(`{"messages":[{"role":"user","content":"intercepted"}]}`)}
			},
		},
	})
	if err != nil {
		t.Fatalf("ExecuteResolved: %v", err)
	}
	if !strings.Contains(gotBody, "intercepted") {
		t.Fatalf("body=%s", gotBody)
	}
	if callerHeaders.Get("X-Resolved-Policy") != "" || callerHeaders.Get("X-Caller") != "preserved" {
		t.Fatalf("caller headers mutated: %#v", callerHeaders)
	}
	if gotTransportAuth == nil || gotTransportAuth.ID != "credential-17" || gotTransportAuth.Metadata["tenant"] != "tenant-1" || gotTransportAuth.Attributes["egress_zone"] != "cn-east" {
		t.Fatalf("transport auth=%#v, want original host identity and metadata", gotTransportAuth)
	}
	if gotIntercept.Model != "deepseek-chat" || gotIntercept.RequestedModel != "canonical-model" || gotIntercept.Metadata["project_id"] != "mutated" {
		t.Fatalf("intercept=%#v", gotIntercept)
	}
	if callerMetadata["project_id"] != "project-1" {
		t.Fatalf("caller metadata mutated: %#v", callerMetadata)
	}
	if candidateAuth.Attributes["base_url"] != "" || candidateAuth.Attributes["egress_zone"] != "cn-east" {
		t.Fatalf("candidate auth mutated: %#v", candidateAuth.Attributes)
	}
}

func TestEmbeddedRuntimeExecuteResolvedInterceptorHandlesNilHeaders(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"ok","choices":[]}`))
	}))
	defer upstream.Close()
	runtime := resolvedTestRuntime(t)
	_, err := runtime.ExecuteResolved(t.Context(), ResolvedExecutionRequest{
		Executor:   NativeExecutorOpenAICompatible,
		Provider:   "deepseek",
		Candidates: []ResolvedExecutionCandidate{{Auth: &coreauth.Auth{Attributes: map[string]string{"api_key": "secret"}}, RuntimeModel: "deepseek-chat", APIBase: upstream.URL}},
		Request:    cliproxyexecutor.Request{Payload: []byte(`{"messages":[]}`), Format: cliproxytranslator.FormatOpenAI},
		Options: cliproxyexecutor.Options{
			SourceFormat:   cliproxytranslator.FormatOpenAI,
			ResponseFormat: cliproxytranslator.FormatOpenAI,
			RequestAfterAuthInterceptor: func(_ context.Context, input cliproxyexecutor.RequestAfterAuthInterceptRequest) cliproxyexecutor.RequestAfterAuthInterceptResponse {
				return cliproxyexecutor.RequestAfterAuthInterceptResponse{Headers: http.Header{"X-Test": []string{"value"}}}
			},
		},
	})
	if err != nil {
		t.Fatalf("ExecuteResolved: %v", err)
	}
}

func TestEmbeddedRuntimeResolvedCandidatesUseUniqueFallbackAuthIDs(t *testing.T) {
	runtime := resolvedTestRuntime(t)
	request := ResolvedExecutionRequest{
		Executor: NativeExecutorOpenAICompatible,
		Provider: "deepseek",
		Candidates: []ResolvedExecutionCandidate{{
			Auth:         &coreauth.Auth{Attributes: map[string]string{"api_key": "secret"}},
			RuntimeModel: "deepseek-chat",
			APIBase:      "https://example.invalid",
		}},
	}
	_, _, first, err := runtime.prepareResolvedCandidates(request)
	if err != nil {
		t.Fatalf("first prepare: %v", err)
	}
	_, _, second, err := runtime.prepareResolvedCandidates(request)
	if err != nil {
		t.Fatalf("second prepare: %v", err)
	}
	if first[0].Auth.ID == "" || first[0].Auth.ID == second[0].Auth.ID {
		t.Fatalf("fallback auth IDs=%q/%q, want unique non-empty IDs", first[0].Auth.ID, second[0].Auth.ID)
	}
	if first[0].TransportAuth.ID != first[0].Auth.ID {
		t.Fatalf("transport auth ID=%q, want execution auth ID %q", first[0].TransportAuth.ID, first[0].Auth.ID)
	}
}

type roundTripperProviderFunc func(*coreauth.Auth) http.RoundTripper

func (f roundTripperProviderFunc) RoundTripperFor(auth *coreauth.Auth) http.RoundTripper {
	return f(auth)
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }

func resolvedTestRuntime(t *testing.T) *EmbeddedRuntime {
	t.Helper()
	manager := coreauth.NewManager(nil, nil, nil)
	runtime, err := NewEmbeddedRuntime(&config.Config{}, manager)
	if err != nil {
		t.Fatalf("NewEmbeddedRuntime: %v", err)
	}
	if err := runtime.Start(t.Context()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	return runtime
}
