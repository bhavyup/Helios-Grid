import { AuthGate } from "@/components/auth-gate";
import { DashboardContent } from "@/components/dashboard-content";

export default function DashboardPage(): JSX.Element {
  return (
    <AuthGate>
      <DashboardContent />
    </AuthGate>
  );
}
