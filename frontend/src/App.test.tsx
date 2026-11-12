import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

function response(payload: unknown, ok = true) {
  return {
    ok,
    json: async () => payload,
    text: async () => JSON.stringify(payload),
  } as Response;
}

describe("dashboard interactions", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const path = String(input);
        if (path === "/api/health") return response({ status: "ok", environment: "test", version: "0.1.0" });
        if (path === "/api/datasets") return response([]);
        if (path === "/api/buildings") return response([{ building_id: "academic", name: "Academic", category: "academic", datasets: ["i_blend"] }]);
        if (path === "/api/experiments") return response([]);
        if (path === "/api/jobs/job-1") return response({ job_id: "job-1", status: "completed", result: {} });
        if (init?.method === "POST" && path === "/api/energy/predict") return response({ job_id: "job-1", status: "queued", message: "queued" });
        if (init?.method === "POST" && path === "/api/datasets/audit") return response({ research_ready: false, coverage_by_kind: {} });
        return response({ detail: `Unhandled test request: ${path}` }, false);
      }),
    );
  });

  it("submits an energy forecast with explicit measurement semantics", async () => {
    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: /Energy forecasting/ }));
    fireEvent.click(screen.getByRole("button", { name: /Generate forecast/ }));
    await waitFor(() => {
      const calls = vi.mocked(fetch).mock.calls;
      const call = calls.find(([input]) => String(input) === "/api/energy/predict");
      expect(call).toBeDefined();
      const body = JSON.parse(String(call?.[1]?.body));
      expect(body.energy_measurement.source_unit).toBe("W");
      expect(body.energy_measurement.semantics).toBe("instantaneous_power");
    });
  });

  it("runs the dataset audit control and reports the result", async () => {
    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: /Dataset explorer/ }));
    fireEvent.click(screen.getByRole("button", { name: /Audit dataset/ }));
    expect(await screen.findByText(/energy semantics still need resolution/i)).toBeTruthy();
  });
});
