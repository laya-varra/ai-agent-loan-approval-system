import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "react-hot-toast";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";

const LoanForm = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    no_of_dependents: "",
    education: "Graduate",
    self_employed: "No",
    income_annum: "",
    loan_amount: "",
    loan_term: "",
    cibil_score: "",
    residential_assets_value: "",
    commercial_assets_value: "",
    luxury_assets_value: "",
    bank_asset_value: "",
  });

  const [loading, setLoading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  // For Select components
  const handleSelectChange = (field: string, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Basic validation
    const requiredFields = [
      "no_of_dependents",
      "education",
      "self_employed",
      "income_annum",
      "loan_amount",
      "loan_term",
      "cibil_score",
    ];
    for (const field of requiredFields) {
      if (!formData[field as keyof typeof formData]) {
        toast.error(`Please fill in ${field.replace(/_/g, " ")}`);
        return;
      }
    }

    setLoading(true);

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      let result;
      try {
        result = await response.json();
      } catch (jsonErr) {
        toast.error("Invalid server response.");
        setLoading(false);
        console.error("JSON parse error:", jsonErr);
        return;
      }

      console.log("API Response:", result);

      if (!response.ok) {
        toast.error(result.error || "Failed to process loan application");
        setLoading(false);
        return;
      }

      toast.success("Application submitted!");
      navigate("/loan-results", { state: { result, formData } });
    } catch (error) {
      toast.error("Network error. Please try again.");
      console.error("Network error:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-violet-100 flex items-center justify-center">
      <Card className="w-full max-w-2xl shadow-lg">
        <CardHeader>
          <CardTitle className="text-2xl font-bold text-center">
            Loan Application Form
          </CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label>Number of Dependents</Label>
                <Input
                  type="number"
                  name="no_of_dependents"
                  value={formData.no_of_dependents}
                  onChange={handleChange}
                  min="0"
                  required
                />
              </div>
              <div>
                <Label>Education</Label>
                <Select
                  value={formData.education}
                  onValueChange={(value) =>
                    handleSelectChange("education", value)
                  }
                  required
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select education" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Graduate">Graduate</SelectItem>
                    <SelectItem value="Not Graduate">Not Graduate</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label>Self Employed</Label>
                <Select
                  value={formData.self_employed}
                  onValueChange={(value) =>
                    handleSelectChange("self_employed", value)
                  }
                  required
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select self employed" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="No">No</SelectItem>
                    <SelectItem value="Yes">Yes</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label>Annual Income</Label>
                <Input
                  type="number"
                  name="income_annum"
                  value={formData.income_annum}
                  onChange={handleChange}
                  min="0"
                  required
                />
              </div>
              <div>
                <Label>Loan Amount</Label>
                <Input
                  type="number"
                  name="loan_amount"
                  value={formData.loan_amount}
                  onChange={handleChange}
                  min="0"
                  required
                />
              </div>
              <div>
                <Label>Loan Term (months)</Label>
                <Input
                  type="number"
                  name="loan_term"
                  value={formData.loan_term}
                  onChange={handleChange}
                  min="0"
                  required
                />
              </div>
              <div>
                <Label>CIBIL Score</Label>
                <Input
                  type="number"
                  name="cibil_score"
                  value={formData.cibil_score}
                  onChange={handleChange}
                  min="300"
                  max="900"
                  required
                />
              </div>
              <div>
                <Label>Residential Assets Value (optional)</Label>
                <Input
                  type="number"
                  name="residential_assets_value"
                  value={formData.residential_assets_value}
                  onChange={handleChange}
                  min="0"
                />
              </div>
              <div>
                <Label>Commercial Assets Value (optional)</Label>
                <Input
                  type="number"
                  name="commercial_assets_value"
                  value={formData.commercial_assets_value}
                  onChange={handleChange}
                  min="0"
                />
              </div>
              <div>
                <Label>Luxury Assets Value (optional)</Label>
                <Input
                  type="number"
                  name="luxury_assets_value"
                  value={formData.luxury_assets_value}
                  onChange={handleChange}
                  min="0"
                />
              </div>
              <div>
                <Label>Bank Asset Value (optional)</Label>
                <Input
                  type="number"
                  name="bank_asset_value"
                  value={formData.bank_asset_value}
                  onChange={handleChange}
                  min="0"
                />
              </div>
            </div>
            <Button
              type="submit"
              className="w-full bg-purple-500 hover:bg-purple-600"
              disabled={loading}
            >
              {loading ? "Submitting..." : "Submit Application"}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
};

export default LoanForm;
