// pages/api/forward-to-n8n.js

export  async function POST(req) {
    if (req.method !== 'POST') {
        return Response.json({message:"ONLY POST METHOD ACCEPTED"})
  }

  try {
    let req_body = await req.json()
    console.log('HEY :',req_body)
    const response = await fetch('https://achref888.app.n8n.cloud/webhook/1d30ace0-f76d-47f0-8860-9e6d94d80cdc', {
        method: 'POST',
        
        body: JSON.stringify(req_body),
    });
    
    console.log('HEY :',response)
    const data = await response.json();
    console.log('data : ',data)
    return Response.json({ data })

    }
    catch (err) {
    return Response.json({ err })

    }
}
